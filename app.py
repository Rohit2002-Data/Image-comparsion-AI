import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
import webcolors
import json

# ---------------- Utilities ----------------
def img_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def avg_color_bgr(img_bgr):
    return np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).reshape(-1,3), axis=0)

def closest_color_name(rgb):
    try:
        return webcolors.rgb_to_name(tuple(int(c) for c in rgb))
    except ValueError:
        min_dist = float('inf')
        closest_name = None
        for hex_val, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = tuple(int(hex_val[i:i+2],16) for i in (1,3,5))
            dist = np.linalg.norm(np.array([r_c,g_c,b_c]) - np.array(rgb))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

def contour_extract(image_bgr, min_area=1500):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,15,9)
    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: continue
        comps.append({'x':x,'y':y,'w':w,'h':h,'crop':image_bgr[y:y+h,x:x+w]})
    return sorted(comps,key=lambda c:c['w']*c['h'],reverse=True)

# ---------------- StateGraph ----------------
END="__END__"

class StateGraph:
    def __init__(self):
        self.nodes={}
        self.edges={}
        self.entry=None
    def add_node(self,name,fn): self.nodes[name]=fn; self.edges.setdefault(name,[])
    def add_edge(self,src,dst): self.edges.setdefault(src,[]).append(dst)
    def set_entry_point(self,name): self.entry=name
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

# ---------------- CLIP + LoRA model ----------------
@st.cache_resource
def load_clip_lora_model():
    device="cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # LoRA adapter (example on visual.proj)
    config=LoraConfig(r=8,lora_alpha=16.0,target_modules=["visual.proj"],
                      lora_dropout=0.0,bias="none",task_type="IMAGE_CLASSIFICATION")
    peft_model = get_peft_model(clip_model,config)
    peft_model.eval()
    return peft_model, processor, device

def get_embedding(crop, model, processor, device):
    img=Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().squeeze()

# ---------------- Nodes ----------------
def node_extract_figma(state):
    state['figma_components'] = contour_extract(state['figma_img'])
    return state

def node_extract_website(state):
    state['website_components'] = contour_extract(state['website_img'])
    return state

# ---------------- Optimized Compare Node ----------------
def node_compare_mismatch(state):
    model, processor, device = load_clip_lora_model()
    
    # Compute Figma embeddings once
    if 'figma_embeddings' not in state:
        fig_embs=[]
        for f in state['figma_components']:
            emb=get_embedding(f['crop'], model, processor, device)
            fig_embs.append({'node':f,'embedding':emb})
        state['figma_embeddings']=fig_embs
    else:
        fig_embs=state['figma_embeddings']

    mismatches=[]
    web_crops=[wb['crop'] for wb in state['website_components']]
    web_coords=[(wb['x'],wb['y'],wb['w'],wb['h']) for wb in state['website_components']]

    # Batch embeddings for website components
    web_embs=[]
    for crop in web_crops:
        emb=get_embedding(crop, model, processor, device)
        web_embs.append(emb)

    for wb_idx, wb_emb in enumerate(web_embs):
        wb = state['website_components'][wb_idx]
        best=None; best_d=float('inf')
        for f in fig_embs:
            d=np.linalg.norm(f['embedding'] - wb_emb)
            if d<best_d:
                best_d=d
                best=f['node']
        fig_w,fig_h=best['w'],best['h']
        web_w,web_h=wb['w'],wb['h']
        status_layout="match" if abs(fig_w-web_w)<=10 and abs(fig_h-web_h)<=10 else "mismatch"

        fig_color=np.round(avg_color_bgr(best['crop'])).astype(int)
        web_color=np.round(avg_color_bgr(wb['crop'])).astype(int)
        status_color="match" if np.linalg.norm(fig_color-web_color)<=30 else "mismatch"

        if status_layout=="mismatch" or status_color=="mismatch":
            mismatches.append({
                'figma_dim':f"{fig_w} x {fig_h}",
                'website_dim':f"{web_w} x {web_h}",
                'status_layout':status_layout,
                'color_figma':closest_color_name(fig_color),
                'color_website':closest_color_name(web_color),
                'status_color':status_color,
                'web_coords':web_coords[wb_idx]
            })

    state['mismatches']=mismatches
    return state


def node_draw_overlay(state):
    overlay = state['website_img'].copy()
    for m in state['mismatches']:
        x,y,w,h = m['web_coords']
        layout = (m['status_layout']=="mismatch")
        color = (m['status_color']=="mismatch")
        if layout and not color:
            box_color=(0,0,255) # Red
        elif not layout and color:
            box_color=(255,0,0) # Blue
        else:
            box_color=(128,0,128) # Purple
        cv2.rectangle(overlay,(x,y),(x+w,y+h),box_color,3)
    # Legend
    cv2.rectangle(overlay,(10,10),(260,110),(255,255,255),-1)
    cv2.putText(overlay,'Red → Layout mismatch',(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(overlay,'Blue → Color mismatch',(20,65),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    cv2.putText(overlay,'Purple → Both mismatch',(20,95),cv2.FONT_HERSHEY_SIMPLEX,0.7,(128,0,128),2)
    state['overlay_image']=overlay
    return state

def node_export_json(state):
    state['report_json']=json.dumps(state['mismatches'],indent=2)
    return state

# ---------------- Graph Build ----------------
sg=StateGraph()
sg.add_node('extract_figma',node_extract_figma)
sg.add_node('extract_website',node_extract_website)
sg.add_node('compare_mismatch',node_compare_mismatch)
sg.add_node('draw_overlay',node_draw_overlay)
sg.add_node('export_json',node_export_json)
sg.set_entry_point('extract_figma')
sg.add_edge('extract_figma','extract_website')
sg.add_edge('extract_website','compare_mismatch')
sg.add_edge('compare_mismatch','draw_overlay')
sg.add_edge('draw_overlay','export_json')
sg.add_edge('export_json',END)
executor=sg.compile()

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")
st.title("🚨 Figma ↔ Website Comparator (CLIP + LoRA) — Mismatches Only")

col1,col2=st.columns(2)
fig_file=col1.file_uploader("Upload Figma Image",type=['png','jpg','jpeg'])
web_file=col2.file_uploader("Upload Website Screenshot",type=['png','jpg','jpeg'])

if fig_file and web_file:
    fig_img=Image.open(fig_file)
    web_img=Image.open(web_file)
    fig_cv=img_to_cv(fig_img)
    web_cv=img_to_cv(web_img)
    st.image([fig_img,web_img],caption=["Figma","Website"],width=350)
    if st.button("Run Pipeline"):
        out=executor({'figma_img':fig_cv,'website_img':web_cv})
        if out['mismatches']:
            st.image(cv_to_pil(out['overlay_image']),caption="Mismatches Overlay",use_column_width=True)
            st.subheader("Mismatched Components")
            st.json(out['report_json'])
        else:
            st.success("✅ No mismatches detected — layout & color match!")
