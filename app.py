import streamlit as st
import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from webcolors import rgb_to_name, CSS3_HEX_TO_NAMES

# -------------------- Utilities --------------------
def ensure_dir(d): os.makedirs(d, exist_ok=True)
ensure_dir('./artifacts')

def img_to_cv(img): return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
def cv_to_pil(img_cv): return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

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

def avg_color_bgr(img_bgr):
    return np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).reshape(-1,3),axis=0)

def closest_color_name(rgb):
    try:
        return rgb_to_name(tuple(rgb))
    except:
        min_dist = float('inf')
        closest_name = None
        for hex_val, name in CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = tuple(int(hex_val[i:i+2],16) for i in (1,3,5))
            dist = np.linalg.norm(np.array([r_c,g_c,b_c])-np.array(rgb))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

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

def node_extract_website(state):
    state['website_components'] = contour_extract(state['website_img'])
    state['web_yolo_boxes'] = state['website_components']  # simple contour detection
    return state

def node_pretrain_clip(state):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    state['clip_model'] = model
    state['clip_processor'] = processor
    state['clip_device'] = device

    embeddings = []
    with torch.no_grad():
        for c in state['figma_components']:
            img = Image.fromarray(cv2.cvtColor(c['crop'], cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt").to(device)
            emb = model.get_image_features(**inputs)
            embeddings.append({'node': c, 'embedding': emb.cpu().numpy().squeeze()})
    state['fig_embeddings'] = embeddings
    return state

def node_compare_clip(state):
    model = state['clip_model']
    processor = state['clip_processor']
    device = state['clip_device']
    fig_embs = state['fig_embeddings']
    mismatches = []

    for wb in state['web_yolo_boxes']:
        img = Image.fromarray(cv2.cvtColor(wb['crop'], cv2.COLOR_BGR2RGB))
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            wemb = model.get_image_features(**inputs).cpu().numpy().squeeze()

        # find best matching Figma component
        best = None; best_d = float('inf')
        for f in fig_embs:
            d = np.linalg.norm(f['embedding'] - wemb)
            if d < best_d:
                best_d = d
                best = f['node']

        # layout check
        fig_w, fig_h = best['w'], best['h']
        web_w, web_h = wb['w'], wb['h']
        status_layout = "match" if abs(fig_w-web_w)<=10 and abs(fig_h-web_h)<=10 else "mismatch"

        # color check
        fig_color = np.round(avg_color_bgr(best['crop'])).astype(int)
        web_color = np.round(avg_color_bgr(wb['crop'])).astype(int)
        status_color = "match" if np.linalg.norm(fig_color-web_color)<=30 else "mismatch"

        if status_layout=="mismatch" or status_color=="mismatch":
            mismatches.append({
                'figma_dim': f"{fig_w} x {fig_h}",
                'website_dim': f"{web_w} x {web_h}",
                'status_layout': status_layout,
                'color_figma': closest_color_name(fig_color),
                'color_website': closest_color_name(web_color),
                'status_color': status_color,
                'web_coords': (wb['x'], wb['y'], wb['w'], wb['h'])
            })

    state['mismatches'] = mismatches
    return state

def node_overlay(state):
    web = state['website_img'].copy()
    for m in state.get('mismatches', []):
        x,y,w,h = m['web_coords']
        layout = m['status_layout']=="mismatch"
        color = m['status_color']=="mismatch"
        if layout and color:
            box_color = (128,0,128) # purple
        elif layout:
            box_color = (0,0,255)   # red
        elif color:
            box_color = (255,0,0)   # blue
        else: continue
        cv2.rectangle(web,(x,y),(x+w,y+h),box_color,3)

    # Legend
    cv2.rectangle(web,(10,10),(220,100),(255,255,255),-1)
    cv2.putText(web,"Red: Layout mismatch",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(web,"Blue: Color mismatch",(20,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(web,"Purple: Both mismatch",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(128,0,128),2)

    state['overlay_image'] = web
    return state

# -------------------- Build StateGraph --------------------
sg = StateGraph()
sg.add_node("extract_figma", node_extract_figma)
sg.add_node("pretrain_clip", node_pretrain_clip)
sg.add_node("extract_website", node_extract_website)
sg.add_node("compare", node_compare_clip)
sg.add_node("overlay", node_overlay)
sg.set_entry_point("extract_figma")
sg.add_edge("extract_figma","pretrain_clip")
sg.add_edge("pretrain_clip","extract_website")
sg.add_edge("extract_website","compare")
sg.add_edge("compare","overlay")
sg.add_edge("overlay", END)
executor = sg.compile()

# -------------------- Streamlit UI --------------------
st.title("Figma ↔ Website Comparator (CLIP + LoRA Upgrade)")

col1, col2 = st.columns(2)
fig_file = col1.file_uploader("Upload Figma Image", type=['png','jpg'])
web_file = col2.file_uploader("Upload Website Screenshot", type=['png','jpg'])

if fig_file and web_file:
    fig_img = Image.open(fig_file)
    web_img = Image.open(web_file)
    fig_cv = img_to_cv(fig_img)
    web_cv = img_to_cv(web_img)
    st.image([fig_img, web_img], caption=["Figma","Website"], width=350)

    if st.button("Run Comparison"):
        state = {"figma_img": fig_cv, "website_img": web_cv}
        out = executor(state)

        st.image(cv_to_pil(out['overlay_image']), caption="Overlay Mismatches")
        if out.get("mismatches"):
            st.subheader("Mismatched Components")
            st.json(out['mismatches'])
        else:
            st.success("No mismatches detected ✅")
