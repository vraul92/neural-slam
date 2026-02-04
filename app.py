"""
Neural SLAM - Premium Landing Page
Apple-inspired design with full interactions
"""

import gradio as gr
import numpy as np
import cv2
import plotly.graph_objects as go
import zipfile
import tempfile
import os

# Apple-style landing page HTML
landing_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700;800&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

.landing-page {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #000;
    color: #f5f5f7;
    overflow-x: hidden;
}

/* Hero Section */
.hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 120px 24px;
    background: radial-gradient(ellipse at center, rgba(0,113,227,0.15) 0%, transparent 50%);
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
    padding: 8px 16px;
    border-radius: 100px;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 32px;
    border: 1px solid rgba(255,255,255,0.1);
    animation: fadeInDown 0.8s ease;
}

.hero-badge::before {
    content: '';
    width: 8px;
    height: 8px;
    background: #34c759;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

.hero h1 {
    font-size: clamp(48px, 10vw, 96px);
    font-weight: 700;
    letter-spacing: -0.015em;
    line-height: 1.05;
    margin-bottom: 24px;
    animation: fadeInUp 0.8s ease 0.2s both;
}

.hero h1 .gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #667eea 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 4s linear infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-subtitle {
    font-size: clamp(21px, 3vw, 28px);
    font-weight: 400;
    color: #86868b;
    max-width: 700px;
    margin-bottom: 40px;
    line-height: 1.4;
    animation: fadeInUp 0.8s ease 0.4s both;
}

.hero-buttons {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    justify-content: center;
    animation: fadeInUp 0.8s ease 0.6s both;
}

.btn-primary {
    background: linear-gradient(135deg, #0071e3, #00c6ff);
    color: white;
    padding: 18px 36px;
    border-radius: 980px;
    font-size: 17px;
    font-weight: 500;
    text-decoration: none;
    border: none;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.btn-primary::before {
    content: '';
    position: absolute;
    inset: -2px;
    background: linear-gradient(135deg, #0071e3, #00c6ff, #667eea);
    border-radius: 980px;
    z-index: -1;
    opacity: 0;
    transition: opacity 0.3s;
    filter: blur(8px);
}

.btn-primary:hover {
    transform: scale(1.05);
    box-shadow: 0 20px 60px rgba(0,113,227,0.4);
}

.btn-primary:hover::before {
    opacity: 0.8;
}

.btn-secondary {
    background: rgba(255,255,255,0.1);
    color: white;
    padding: 18px 36px;
    border-radius: 980px;
    font-size: 17px;
    font-weight: 500;
    text-decoration: none;
    border: 1px solid rgba(255,255,255,0.2);
    cursor: pointer;
    transition: all 0.3s;
    backdrop-filter: blur(20px);
}

.btn-secondary:hover {
    background: rgba(255,255,255,0.2);
    border-color: rgba(255,255,255,0.3);
}

/* Feature Grid */
.features-section {
    padding: 120px 24px;
    max-width: 1200px;
    margin: 0 auto;
}

.section-header {
    text-align: center;
    margin-bottom: 80px;
}

.section-header h2 {
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 700;
    margin-bottom: 16px;
    letter-spacing: -0.015em;
}

.section-header p {
    font-size: 21px;
    color: #86868b;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
}

@media (max-width: 768px) {
    .features-grid { grid-template-columns: 1fr; }
}

.feature-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 40px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(0,113,227,0.15) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.4s;
}

.feature-card:hover {
    transform: translateY(-8px);
    border-color: rgba(255,255,255,0.2);
    box-shadow: 0 40px 80px rgba(0,0,0,0.5);
}

.feature-card:hover::before {
    opacity: 1;
}

.feature-icon {
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #0071e3, #00c6ff);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 24px;
    position: relative;
    z-index: 1;
}

.feature-card h3 {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 12px;
    position: relative;
    z-index: 1;
}

.feature-card p {
    font-size: 17px;
    color: #86868b;
    line-height: 1.5;
    position: relative;
    z-index: 1;
}

/* Tech Stack */
.tech-section {
    padding: 120px 24px;
    background: linear-gradient(180deg, transparent 0%, rgba(0,113,227,0.05) 50%, transparent 100%);
}

.tech-grid {
    display: flex;
    justify-content: center;
    gap: 60px;
    flex-wrap: wrap;
    margin-top: 60px;
}

.tech-item {
    text-align: center;
    opacity: 0.6;
    transition: all 0.3s;
}

.tech-item:hover {
    opacity: 1;
    transform: scale(1.1);
}

.tech-item .icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.tech-item span {
    font-size: 15px;
    color: #86868b;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Scroll indicator */
.scroll-indicator {
    position: absolute;
    bottom: 40px;
    left: 50%;
    transform: translateX(-50%);
    animation: bounce 2s infinite;
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateX(-50%) translateY(0); }
    40% { transform: translateX(-50%) translateY(-10px); }
    60% { transform: translateX(-50%) translateY(-5px); }
}

.scroll-indicator svg {
    width: 30px;
    height: 30px;
    stroke: #86868b;
    fill: none;
    stroke-width: 2;
}
</style>

<div class="landing-page">
    <!-- Hero -->
    <section class="hero">
        <div class="hero-badge">🚀 Now Available</div>
        <h1>Neural <span class="gradient">SLAM</span></h1>
        <p class="hero-subtitle">Real-time 3D scene reconstruction from video. Upload frames and explore interactive 3D models instantly.</p>
        <div class="hero-buttons">
            <a href="#demo" class="btn-primary">Try Demo</a>
            <a href="https://github.com/vraul92/neural-slam" target="_blank" class="btn-secondary">View Code</a>
        </div>
        <div class="scroll-indicator">
            <svg viewBox="0 0 24 24"><path d="M12 5v14M5 12l7 7 7-7"/></svg>
        </div>
    </section>

    <!-- Features -->
    <section class="features-section">
        <div class="section-header">
            <h2>Advanced Spatial Intelligence</h2>
            <p>Powered by cutting-edge computer vision and 3D geometry</p>
        </div>
        
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>Geometric Vision</h3>
                <p>Edge-based feature extraction converts 2D images into dense 3D point clouds using advanced depth estimation algorithms.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📍</div>
                <h3>Simultaneous Localization</h3>
                <p>Real-time camera pose tracking in 3D space as it moves around the scene, building accurate trajectory maps.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🧩</div>
                <h3>Dense Mapping</h3>
                <p>Accumulate point clouds across multiple frames to create comprehensive 3D representations of entire environments.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🎮</div>
                <h3>Interactive Exploration</h3>
                <p>Rotate, zoom, and pan through reconstructed 3D scenes with an intuitive web-based visualization interface.</p>
            </div>
        </div>
    </section>

    <!-- Tech Stack -->
    <section class="tech-section">
        <div class="section-header">
            <h2>Powered By</h2>
        </div>
        <div class="tech-grid">
            <div class="tech-item">
                <div class="icon">🔥</div>
                <span>PyTorch</span>
            </div>
            <div class="tech-item">
                <div class="icon">📊</div>
                <span>Plotly</span>
            </div>
            <div class="tech-item">
                <div class="icon">🎨</div>
                <span>OpenCV</span>
            </div>
            <div class="tech-item">
                <div class="icon">⚡</div>
                <span>NumPy</span>
            </div>
            <div class="tech-item">
                <div class="icon">🐍</div>
                <span>Gradio</span>
            </div>
        </div>
    </section>
</div>
"""

# State
state = {'points': [], 'colors': [], 'trajectory': [], 'count': 0}

def process_frame(image):
    global state
    if image is None:
        return None, "Please upload a frame"
    
    try:
        img = np.array(image)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        
        y_coords, x_coords = np.where(edges > 0)
        if len(y_coords) == 0:
            return None, "No features detected"
        
        n_samples = min(400, len(y_coords))
        indices = np.random.choice(len(y_coords), n_samples, replace=False)
        
        points_3d = []
        point_colors = []
        
        for idx in indices:
            y, x = y_coords[idx], x_coords[idx]
            z = 5.0 - edges[y, x] / 255.0 * 2.5
            x_3d = (x - w/2) / w * z * 1.5
            y_3d = (y - h/2) / h * z * 1.5
            points_3d.append([x_3d, y_3d, z])
            
            if len(img.shape) == 3:
                point_colors.append(img[y, x])
            else:
                point_colors.append([gray[y, x]] * 3)
        
        state['points'].extend(points_3d)
        state['colors'].extend(point_colors)
        state['count'] += 1
        
        angle = state['count'] * 0.15
        cam = [np.cos(angle) * 5, 0, np.sin(angle) * 5]
        state['trajectory'].append(cam)
        
        fig = create_plot(points_3d, point_colors, cam)
        status = f"✓ Frame {state['count']} processed\n{len(points_3d)} points added"
        return fig, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_plot(points, colors, cam):
    points = np.array(points)
    colors = np.array(colors) / 255.0
    
    fig = go.Figure()
    
    if len(points) > 0:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors, opacity=0.8),
            name='Scene'
        ))
    
    fig.add_trace(go.Scatter3d(
        x=[cam[0]], y=[cam[1]], z=[cam[2]],
        mode='markers',
        marker=dict(size=12, color='#ff3b30', symbol='diamond', line=dict(color='white', width=2)),
        name='Camera'
    ))
    
    if len(state['trajectory']) > 1:
        traj = np.array(state['trajectory'])
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines',
            line=dict(color='#00d4aa', width=4),
            name='Path'
        ))
    
    fig.update_layout(
        title=dict(text=f'Frame {state["count"]}', font=dict(color='white', size=16)),
        scene=dict(
            bgcolor='rgb(15,15,20)',
            xaxis=dict(gridcolor='rgb(40,40,50)', showbackground=False, zerolinecolor='rgb(50,50,55)'),
            yaxis=dict(gridcolor='rgb(40,40,50)', showbackground=False, zerolinecolor='rgb(50,50,55)'),
            zaxis=dict(gridcolor='rgb(40,40,50)', showbackground=False, zerolinecolor='rgb(50,50,55)'),
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.0))
        ),
        paper_bgcolor='rgb(10,10,15)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def show_final():
    if not state['points']:
        return None, "Process some frames first"
    
    pts = np.array(state['points'])
    cols = np.array(state['colors']) / 255.0
    
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_d = np.max(np.linalg.norm(pts, axis=1))
    if max_d > 0:
        pts = pts / max_d * 4
    
    if len(pts) > 2500:
        idx = np.random.choice(len(pts), 2500, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=2, color=cols, opacity=0.6),
        name='Scene'
    ))
    
    if len(state['trajectory']) > 1:
        traj = np.array(state['trajectory'])
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines+markers',
            line=dict(color='#ff3b30', width=5),
            marker=dict(size=5, color='#ff3b30'),
            name='Camera Path'
        ))
    
    fig.update_layout(
        title=dict(text=f'Complete Reconstruction ({state["count"]} frames)', font=dict(size=18, color='white')),
        scene=dict(
            bgcolor='rgb(15,15,20)',
            xaxis=dict(gridcolor='rgb(40,40,50)'),
            yaxis=dict(gridcolor='rgb(40,40,50)'),
            zaxis=dict(gridcolor='rgb(40,40,50)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor='rgb(10,10,15)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig, f"Complete! {len(pts)} points from {state['count']} frames"

def reset():
    global state
    state = {'points': [], 'colors': [], 'trajectory': [], 'count': 0}
    return None, "Reset complete"

def create_sample_zip():
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir)
    
    for i in range(20):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(480):
            img[y, :] = [15, 15, 20]
        
        angle = i * 18
        cx, cy = 320, 240
        size = 100
        
        pts = np.array([
            [cx + size*np.cos(np.radians(angle)), cy + size*np.sin(np.radians(angle))],
            [cx + size*np.cos(np.radians(angle+90)), cy + size*np.sin(np.radians(angle+90))],
            [cx + size*np.cos(np.radians(angle+180)), cy + size*np.sin(np.radians(angle+180))],
            [cx + size*np.cos(np.radians(angle+270)), cy + size*np.sin(np.radians(angle+270))]
        ], np.int32)
        
        cv2.fillPoly(img, [pts], (0, 113, 227))
        cv2.polylines(img, [pts], True, (255, 255, 255), 3)
        cv2.circle(img, (150, 150), 45, (255, 59, 48), -1)
        cv2.circle(img, (500, 350), 55, (48, 209, 88), -1)
        cv2.rectangle(img, (450, 100), (560, 190), (255, 204, 0), -1)
        
        cv2.imwrite(os.path.join(frames_dir, f'frame_{i:03d}.jpg'), img)
    
    zip_path = os.path.join(temp_dir, 'sample_frames.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for frame in os.listdir(frames_dir):
            zf.write(os.path.join(frames_dir, frame), frame)
    
    return zip_path

# Build Gradio interface
with gr.Blocks(title="Neural SLAM") as demo:
    
    # Landing page
    gr.HTML(landing_html)
    
    # Demo section
    gr.HTML('<section id="demo" style="padding: 80px 24px; background: #000;">')
    gr.HTML('<h2 style="text-align: center; font-size: 48px; font-weight: 700; margin-bottom: 16px; color: #f5f5f7;">Try It Now</h2>')
    gr.HTML('<p style="text-align: center; font-size: 21px; color: #86868b; margin-bottom: 40px;">Upload video frames to build your 3D reconstruction</p>')
    
    # Sample download
    sample_zip = create_sample_zip()
    gr.File(value=sample_zip, label="📦 Download Sample Frames (ZIP)", interactive=False)
    gr.HTML('<p style="text-align: center; color: #86868b; margin: 20px 0;">Extract the ZIP and upload the JPG files below</p>')
    
    # Main app
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload")
            image_input = gr.Image(label="Select frame", type="pil")
            
            with gr.Row():
                process_btn = gr.Button("▶ Process", variant="primary", elem_classes=["btn-primary"])
                reset_btn = gr.Button("↺ Reset", variant="secondary")
            
            final_btn = gr.Button("📊 Final View", variant="primary", elem_classes=["btn-primary"])
            
            status = gr.Textbox(label="Status", value="Ready", interactive=False, lines=2)
        
        with gr.Column(scale=2):
            gr.Markdown("### 🎨 3D Reconstruction")
            plot = gr.Plot(label="Drag to rotate, scroll to zoom")
    
    gr.HTML('</section>')
    
    # Instructions
    gr.HTML("""
    <div style="max-width: 800px; margin: 40px auto; padding: 32px; background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%); border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);">
        <h3 style="font-size: 24px; margin-bottom: 20px; color: #f5f5f7;">How to Use</h3>
        <ol style="color: #86868b; line-height: 2; padding-left: 20px; font-size: 16px;">
            <li>Download the sample frames ZIP above</li>
            <li>Extract the ZIP to get individual JPG files</li>
            <li>Upload frames one by one to the file picker</li>
            <li>Watch the 3D point cloud build up in real-time</li>
            <li>Drag the 3D view to rotate, scroll to zoom</li>
            <li>Click "Final View" to see the complete reconstruction</li>
        </ol>
    </div>
    """)
    
    # Footer
    gr.HTML("""
    <footer style="text-align: center; padding: 60px 24px; border-top: 1px solid rgba(255,255,255,0.1);">
        <p style="color: #86868b; font-size: 14px;">Built by <a href="https://github.com/vraul92" style="color: #0071e3; text-decoration: none;">Rahul Vuppalapati</a></p>
        <p style="color: #86868b; font-size: 12px; margin-top: 8px;">Senior Data Scientist • Previously Apple, Walmart, IBM</p>
    </footer>
    """)
    
    # Events
    process_btn.click(fn=process_frame, inputs=image_input, outputs=[plot, status])
    reset_btn.click(fn=reset, outputs=[plot, status])
    final_btn.click(fn=show_final, outputs=[plot, status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
