# Previous Version (Before CLIP + LoRA Vision Upgrade)
# --------------------------------------------------
# This version keeps the old lightweight pipeline
# without Vision Transformer, CLIP, or LoRA integration.
# Nodes are simple and rely on contour extraction + small CNN encoder.

import streamlit as st
import os
import io
import json
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any

# -------------------- Utilities --------------------
def ensure_dir(d): os.makedirs(d, exist_ok=True)
ensure_dir('./artifacts')

def img_to_cv(img): return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
def cv_to_pil(img_cv): return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# Simple contour extractor
def contour_extract(image_bgr, min_area=1500):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,9)
    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: continue
        comps.append({'x':x,'y':y,'w':w,'h':h,'crop':image_bgr[y:y+h,x:x+w]})
    return sorted(comps, key=lambda c: c['w']*c['h'], reverse=True)

# -------------------- StateGraph --------------------
END = "__END__"

class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry = None
    def add_node(self, name, fn): self.nodes[name] = fn; self.edges.setdefault(name, [])
    def add_edge(self, src, dst): self.edges.setdefault(src, []).append(dst)
    def set_entry_point(self, name): self.entry = name
    def compile(self):
        def executor(state):
            q=[self.entry]; visited=set()
            while q:
                n=q.pop(0)
                if n==END: break
                if n in visited: continue
                visited.add(n)
                fn=self.nodes.get(n)
                if fn: state=fn(state)
                for nxt in self.edges.get(n,[]):
                    if nxt not in visited: q.append(nxt)
            return state
        return executor

# -------------------- Nodes --------------------
def node_extract_figma(state):
    state['figma_components'] = contour_extract(state['figma_img'])
    return state

# small CNN encoder (no CLIP)
class SmallEnc(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, out)
    def forward(self,x): return self.fc(self.conv(x).view(x.size(0),-1))

def node_pretrain(state):
    enc=SmallEnc(); enc.eval()
    comps = state.get('figma_components',[])
    emb=[]
    with torch.no_grad():
        for c in comps:
            crop=cv2.resize(c['crop'],(64,64))
            t=torch.tensor(crop).permute(2,0,1).float().unsqueeze(0)/255
            emb.append({'node':c,'embedding':enc(t).squeeze(0).numpy().tolist()})
    state['model']=enc
    state['fig_embeddings']=emb
    return state

def node_extract_website(state):
    state['website_components']=contour_extract(state['website_img'])
    return state

def node_align(state):
    state['aligned']=False
    return state

def node_yolo_detect(state):
    state['web_yolo_boxes']=contour_extract(state['website_img'])
    return state

def avg_color_bgr(img_bgr):
    return np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).reshape(-1,3),axis=0)

def node_compare(state):
    fig_embs=state['fig_embeddings']; web_boxes=state['web_yolo_boxes']; model=state['model']
    matches=[]
    for wb in web_boxes:
        crop=cv2.resize(wb['crop'],(64,64))
        t=torch.tensor(crop).permute(2,0,1).float().unsqueeze(0)/255
        with torch.no_grad(): wemb=model(t).squeeze(0).numpy()
        best=None; best_d=1e9
        for f in fig_embs:
            d=np.linalg.norm(np.array(f['embedding'])-wemb)
            if d<best_d: best_d=d; best=f['node']
        dim={'fig_w':best['w'],'fig_h':best['h'],'web_w':wb['w'],'web_h':wb['h']}
        color_diff=np.linalg.norm(avg_color_bgr(best['crop'])-avg_color_bgr(wb['crop']))
        matches.append({'fig':best,'web':wb,'dim':dim,'color_diff':color_diff,'score':best_d})
    state['matches']=matches
    return state

def node_overlay(state):
    web=state['website_img']; overlay=web.copy()
    for m in state['matches']:
        w=m['web']; x,y,wid,hei=w['x'],w['y'],w['w'],w['h']
        is_bad=(abs(m['dim']['fig_w']-wid)>10 or abs(m['dim']['fig_h']-hei)>10)
        color=(0,0,255) if is_bad else (0,255,0)
        cv2.rectangle(overlay,(x,y),(x+wid,y+hei),color,3)
    state['overlay_image']=overlay
    return state

def node_export(state):
    rows=[]
    for m in state['matches']:
        f=m['fig']; w=m['web']
        rows.append({'fig_x':f['x'],'fig_y':f['y'],'fig_w':f['w'],'fig_h':f['h'],
                     'web_x':w['x'],'web_y':w['y'],'web_w':w['w'],'web_h':w['h'],
                     'color_diff':m['color_diff'],'score':m['score']})
    state['report_json']=json.dumps(rows,indent=2)
    state['report_rows']=rows
    return state

# -------------------- Graph Build --------------------
sg=StateGraph()
sg.add_node('extract_figma',node_extract_figma)
sg.add_node('pretrain',node_pretrain)
sg.add_node('extract_website',node_extract_website)
sg.add_node('align',node_align)
sg.add_node('yolo_detect',node_yolo_detect)
sg.add_node('compare',node_compare)
sg.add_node('overlay',node_overlay)
sg.add_node('export',node_export)
sg.set_entry_point('extract_figma')
sg.add_edge('extract_figma','pretrain')
sg.add_edge('pretrain','extract_website')
sg.add_edge('extract_website','align')
sg.add_edge('align','yolo_detect')
sg.add_edge('yolo_detect','compare')
sg.add_edge('compare','overlay')
sg.add_edge('overlay','export')
sg.add_edge('export',END)
executor=sg.compile()

# -------------------- Streamlit UI --------------------
st.title("Figma ↔ Website Comparator — Old Version (No Vision Upgrade)")

col1,col2=st.columns(2)
fig_file=col1.file_uploader('Upload Figma Image',type=['png','jpg'])
web_file=col2.file_uploader('Upload Website Screenshot',type=['png','jpg'])

if fig_file and web_file:
    fig_pil=Image.open(fig_file); web_pil=Image.open(web_file)
    fig_cv=img_to_cv(fig_pil); web_cv=img_to_cv(web_pil)
    st.image([fig_pil,web_pil],caption=['Figma','Website'],width=350)
    if st.button('Run Pipeline'):
        out=executor({'figma_img':fig_cv,'website_img':web_cv})
        st.image(cv_to_pil(out['overlay_image']),caption='Overlay')
        st.code(out['report_json'],language='json')
