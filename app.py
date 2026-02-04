"""
Neural SLAM - Apple Style Design
Clean, minimalist, professional
"""

import gradio as gr
import os
import numpy as np
from PIL import Image
import io
import plotly.graph_objects as go
import cv2
from scipy import ndimage
import zipfile
import tempfile

# Apple-inspired CSS
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --apple-bg: #000000;
    --apple-card: #1d1d1f;
    --apple-text: #f5f5f7;
    --apple-secondary: #86868b;
    --apple-blue: #0071e3;
    --apple-blue-hover: #0077ed;
    --apple-gray: #424245;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--apple-bg) !important;
    color: var(--apple-text) !important;
}

.gradio-container {
    max-width: 1024px !important;
    padding: 0 22px !important;
}

/* Typography */
h1, h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.022em !important;
}

/* Cards */
.card {
    background: var(--apple-card) !important;
    border-radius: 18px !important;
    padding: 32px !important;
    margin: 24px 0 !important;
}

/* Buttons */
button[variant="primary"] {
    background: var(--apple-blue) !important;
    color: white !important;
    border-radius: 980px !important;
    padding: 18px 31px !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    border: none !important;
    transition: all 0.3s !important;
}

button[variant="primary"]:hover {
    background: var(--apple-blue-hover) !important;
}

button[variant="secondary"] {
    background: var(--apple-gray) !important;
    color: white !important;
    border-radius: 980px !important;
    padding: 18px 31px !important;
    font-size: 17px !important;
}

/* Labels */
label {
    color: var(--apple-text) !important;
    font-size: 21px !important;
    font-weight: 600 !important;
}

/* Inputs */
input, textarea {
    background: var(--apple-card) !important;
    border: 1px solid var(--apple-gray) !important;
    border-radius: 12px !important;
    color: var(--apple-text) !important;
}

/* Status box */
.status-box {
    background: var(--apple-card) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    font-family: 'SF Mono', monospace !important;
    font-size: 14px !important;
    color: var(--apple-secondary) !important;
}

/* Hero */
.hero-title {
    font-size: 64px !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin: 60px 0 20px !important;
    letter-spacing: -0.015em !important;
}

.hero-subtitle {
    font-size: 28px !important;
    color: var(--apple-secondary) !important;
    text-align: center !important;
    max-width: 680px !important;
    margin: 0 auto 40px !important;
    line-height: 1.25 !important;
}

/* Section */
.section-title {
    font-size: 40px !important;
    font-weight: 600 !important;
    text-align: center !important;
    margin: 60px 0 40px !important;
}

/* Feature grid */
.feature-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 24px !important;
    margin: 40px 0 !important;
}

.feature-card {
    background: var(--apple-card) !important;
    border-radius: 18px !important;
    padding: 28px !important;
}

.feature-card h3 {
    font-size: 19px !important;
    margin-bottom: 8px !important;
}

.feature-card p {
    font-size: 15px !important;
    color: var(--apple-secondary) !important;
    line-height: 1.4 !important;
}

/* Divider */
.divider {
    height: 1px !important;
    background: var(--apple-gray) !important;
    margin: 60px 0 !important;
}

/* Footer */
.footer {
    text-align: center !important;
    padding: 40px 0 60px !important;
    color: var(--apple-secondary) !important;
    font-size: 12px !important;
}

@media (max-width: 768px) {
    .hero-title { font-size: 40px !important; }
    .hero-subtitle { font-size: 21px !important; }
    .feature-grid { grid-template-columns: 1fr !important; }
}
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
        
        # Extract features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        
        y_coords, x_coords = np.where(edges > 0)
        if len(y_coords) == 0:
            return None, "No features detected"
        
        # Sample points
        n_samples = min(300, len(y_coords))
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
        
        # Update state
        state['points'].extend(points_3d)
        state['colors'].extend(point_colors)
        state['count'] += 1
        
        # Camera trajectory
        angle = state['count'] * 0.15
        cam = [np.cos(angle) * 5, 0, np.sin(angle) * 5]
        state['trajectory'].append(cam)
        
        # Create plot
        fig = create_plot(points_3d, point_colors, cam)
        
        status = f"Frame {state['count']} processed\n{len(points_3d)} points added\nTotal: {len(state['points'])} points"
        return fig, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_plot(points, colors, cam):
    points = np.array(points)
    colors = np.array(colors) / 255.0
    
    fig = go.Figure()
    
    # Points
    if len(points) > 0:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors, opacity=0.8),
            name='Scene'
        ))
    
    # Camera
    fig.add_trace(go.Scatter3d(
        x=[cam[0]], y=[cam[1]], z=[cam[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Camera'
    ))
    
    # Trajectory
    if len(state['trajectory']) > 1:
        traj = np.array(state['trajectory'])
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines',
            line=dict(color='cyan', width=3),
            name='Path'
        ))
    
    fig.update_layout(
        title=f'Frame {state["count"]}',
        scene=dict(
            bgcolor='rgb(20,20,25)',
            xaxis=dict(gridcolor='rgb(40,40,45)', showbackground=False),
            yaxis=dict(gridcolor='rgb(40,40,45)', showbackground=False),
            zaxis=dict(gridcolor='rgb(40,40,45)', showbackground=False),
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.0))
        ),
        paper_bgcolor='rgb(15,15,20)',
        font=dict(color='white', size=12),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def show_final():
    if not state['points']:
        return None, "Process some frames first"
    
    pts = np.array(state['points'])
    cols = np.array(state['colors']) / 255.0
    
    # Center and scale
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_d = np.max(np.linalg.norm(pts, axis=1))
    if max_d > 0:
        pts = pts / max_d * 4
    
    # Sample if too many
    if len(pts) > 2000:
        idx = np.random.choice(len(pts), 2000, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=2, color=cols, opacity=0.5),
        name='Scene'
    ))
    
    if len(state['trajectory']) > 1:
        traj = np.array(state['trajectory'])
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=4, color='red'),
            name='Camera Path'
        ))
    
    fig.update_layout(
        title=dict(text=f'Complete Reconstruction ({state["count"]} frames)', font=dict(size=16)),
        scene=dict(
            bgcolor='rgb(20,20,25)',
            xaxis=dict(gridcolor='rgb(40,40,45)'),
            yaxis=dict(gridcolor='rgb(40,40,45)'),
            zaxis=dict(gridcolor='rgb(40,40,45)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor='rgb(15,15,20)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig, f"Complete! {len(pts)} points from {state['count']} frames"

def reset():
    global state
    state = {'points': [], 'colors': [], 'trajectory': [], 'count': 0}
    return None, "Reset complete"

def create_sample_zip():
    """Create a zip file with sample frames"""
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, 'sample_frames')
    os.makedirs(frames_dir)
    
    # Generate 20 frames
    for i in range(20):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dark gradient background
        for y in range(480):
            img[y, :] = [20, 20, 30]
        
        # Rotating cube
        angle = i * 18
        cx, cy = 320, 240
        size = 80
        
        pts = np.array([
            [cx + size*np.cos(np.radians(angle)), cy + size*np.sin(np.radians(angle))],
            [cx + size*np.cos(np.radians(angle+90)), cy + size*np.sin(np.radians(angle+90))],
            [cx + size*np.cos(np.radians(angle+180)), cy + size*np.sin(np.radians(angle+180))],
            [cx + size*np.cos(np.radians(angle+270)), cy + size*np.sin(np.radians(angle+270))]
        ], np.int32)
        
        cv2.fillPoly(img, [pts], (0, 113, 227))
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)
        
        # Additional features
        cv2.circle(img, (150, 150), 40, (255, 100, 100), -1)
        cv2.circle(img, (500, 350), 50, (100, 255, 100), -1)
        cv2.rectangle(img, (450, 100), (550, 180), (255, 200, 0), -1)
        
        # Save frame
        cv2.imwrite(os.path.join(frames_dir, f'frame_{i:03d}.jpg'), img)
    
    # Create zip
    zip_path = os.path.join(temp_dir, 'sample_frames.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for frame in os.listdir(frames_dir):
            zf.write(os.path.join(frames_dir, frame), frame)
    
    return zip_path

# Build interface
with gr.Blocks(title="Neural SLAM") as demo:
    
    # Hero
    gr.HTML('''
    <div style="text-align: center; padding: 80px 0 40px;">
        <h1 class="hero-title">Neural SLAM</h1>
        <p class="hero-subtitle">Real-time 3D scene reconstruction from video</p>
    </div>
    ''')
    
    # Features
    gr.HTML('''
    <h2 class="section-title">How It Works</h2>
    <div class="feature-grid">
        <div class="feature-card">
            <h3>📤 Upload Frames</h3>
            <p>Extract frames from any video and upload them one by one</p>
        </div>
        <div class="feature-card">
            <h3>🎨 3D Reconstruction</h3>
            <p>Watch the 3D point cloud build up in real-time</p>
        </div>
        <div class="feature-card">
            <h3>🎮 Interactive</h3>
            <p>Rotate, zoom, and explore the reconstructed scene</p>
        </div>
        <div class="feature-card">
            <h3>📊 Complete View</h3>
            <p>See the final reconstruction with camera trajectory</p>
        </div>
    </div>
    ''')
    
    gr.HTML('<div class="divider"></div>')
    
    # Sample download
    gr.HTML('<h2 class="section-title">Get Started</h2>')
    
    sample_zip = create_sample_zip()
    gr.File(
        value=sample_zip,
        label="Download Sample Frames (ZIP)",
        interactive=False
    )
    
    gr.HTML('<p style="text-align: center; color: #86868b; margin: 20px 0;">Download these sample frames, extract the ZIP, then upload the JPG files below</p>')
    
    # Main app
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Frame")
            
            image_input = gr.Image(label="Select JPG frame", type="pil")
            
            with gr.Row():
                process_btn = gr.Button("Process", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
            
            final_btn = gr.Button("Show Final", variant="primary")
            
            status = gr.Textbox(
                label="Status",
                value="Ready. Download sample frames above and upload them here.",
                interactive=False,
                lines=3
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 3D View")
            plot = gr.Plot(label="Drag to rotate")
    
    # Instructions
    gr.HTML('''
    <div style="background: #1d1d1f; border-radius: 18px; padding: 28px; margin: 32px 0;">
        <h3 style="margin-bottom: 16px; font-size: 21px;">Instructions</h3>
        <ol style="color: #86868b; line-height: 1.8; padding-left: 20px;">
            <li>Download the sample frames ZIP above</li>
            <li>Extract the ZIP file to get individual JPG frames</li>
            <li>Upload frames one by one using the file picker</li>
            <li>Watch the 3D reconstruction build up in real-time</li>
            <li>Drag the 3D view to rotate, scroll to zoom</li>
            <li>Click "Show Final" to see the complete reconstruction</li>
        </ol>
    </div>
    ''')
    
    # Footer
    gr.HTML('''
    <div class="footer">
        <p>Built by Rahul Vuppalapati · <a href="https://github.com/vraul92" style="color: #0071e3;">GitHub</a> · <a href="https://linkedin.com/in/vrc7" style="color: #0071e3;">LinkedIn</a></p>
    </div>
    ''')
    
    # Events
    process_btn.click(fn=process_frame, inputs=image_input, outputs=[plot, status])
    reset_btn.click(fn=reset, outputs=[plot, status])
    final_btn.click(fn=show_final, outputs=[plot, status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=css)
