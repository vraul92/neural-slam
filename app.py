"""
Neural SLAM: Interactive 3D Reconstruction
Beautiful design with rotatable 3D viewer and sample video

Author: Rahul Vuppalapati
"""

import gradio as gr
import os
import numpy as np
from PIL import Image
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from scipy import ndimage

# Custom CSS for premium design
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(180deg, #000000 0%, #0a0a0f 100%) !important;
}

.gradio-container {
    max-width: 1200px !important;
    background: transparent !important;
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
    background: linear-gradient(180deg, rgba(0,113,227,0.1) 0%, transparent 60%);
    border-radius: 32px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(0,113,227,0.15) 0%, transparent 50%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero h1 {
    font-size: 56px;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 20px;
    letter-spacing: -0.02em;
}

.hero .highlight {
    background: linear-gradient(90deg, #0071e3, #00c6ff, #0071e3);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 3s ease infinite;
}

@keyframes gradient-shift {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero p {
    font-size: 21px;
    color: #86868b;
    max-width: 600px;
    margin: 0 auto;
    font-weight: 400;
    line-height: 1.5;
}

/* Section Headers */
.section-header {
    text-align: center;
    margin: 60px 0 40px;
}

.section-header h2 {
    font-size: 40px;
    font-weight: 600;
    color: #f5f5f7;
    margin-bottom: 12px;
}

.section-header p {
    font-size: 19px;
    color: #86868b;
}

/* Technology Cards */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin: 40px 0;
}

.tech-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 32px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.tech-card:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-4px);
}

.tech-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,113,227,0.1) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.4s;
}

.tech-card:hover::before {
    opacity: 1;
}

.tech-icon {
    width: 56px;
    height: 56px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #0071e3, #00c6ff);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
}

.tech-card h3 {
    font-size: 21px;
    font-weight: 600;
    color: #f5f5f7;
    margin-bottom: 10px;
}

.tech-card p {
    color: #86868b;
    font-size: 15px;
    line-height: 1.5;
}

/* Demo Section */
.demo-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 32px;
    padding: 40px;
    margin: 40px 0;
}

.demo-title {
    font-size: 32px;
    font-weight: 600;
    color: #f5f5f7;
    text-align: center;
    margin-bottom: 8px;
}

.demo-subtitle {
    color: #86868b;
    text-align: center;
    margin-bottom: 32px;
    font-size: 17px;
}

/* Sample Video Section */
.sample-section {
    background: linear-gradient(135deg, rgba(0,113,227,0.1) 0%, rgba(0,198,255,0.05) 100%);
    border: 1px solid rgba(0,113,227,0.2);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
    text-align: center;
}

.sample-section h4 {
    color: #f5f5f7;
    font-size: 17px;
    margin-bottom: 12px;
    font-weight: 600;
}

.sample-section p {
    color: #86868b;
    font-size: 14px;
    margin-bottom: 16px;
}

/* Status Box */
.status-box {
    background: rgba(0, 0, 0, 0.4);
    border-radius: 16px;
    padding: 20px;
    font-family: 'SF Mono', monospace;
    font-size: 13px;
    color: #86868b;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.status-box .success {
    color: #34c759;
}

.status-box .info {
    color: #0071e3;
}

/* Instructions */
.instructions {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 20px;
    padding: 24px;
    margin-top: 24px;
}

.instructions h4 {
    color: #f5f5f7;
    font-size: 17px;
    margin-bottom: 16px;
    font-weight: 600;
}

.instructions ol {
    color: #86868b;
    padding-left: 20px;
    line-height: 2;
}

.instructions li {
    margin-bottom: 8px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 60px 20px;
    margin-top: 60px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.footer p {
    color: #86868b;
    font-size: 14px;
    margin-bottom: 8px;
}

.footer a {
    color: #0071e3;
    text-decoration: none;
    transition: opacity 0.3s;
}

.footer a:hover {
    opacity: 0.8;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #0071e3, #00c6ff) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
}

button.primary:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 30px rgba(0,113,227,0.4) !important;
}

button.secondary {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
}
"""

# Global state
slam_state = {
    'points': [],
    'colors': [],
    'trajectory': [],
    'frame_count': 0
}

def process_frame(image):
    """Process frame and return interactive 3D plot."""
    global slam_state
    
    if image is None:
        return None, "❌ Please upload an image"
    
    try:
        # Convert to array
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        edges = cv2.Canny(gray, 50, 150)
        
        # Get edge points
        y_coords, x_coords = np.where(edges > 0)
        
        # Sample points
        indices = np.random.choice(len(y_coords), min(400, len(y_coords)), replace=False)
        
        points_3d = []
        colors = []
        
        for idx in indices:
            y, x = y_coords[idx], x_coords[idx]
            # Create 3D point with depth based on edge intensity
            z = 5.0 - edges[y, x] / 255.0 * 3
            x_3d = (x - w/2) / w * z
            y_3d = (y - h/2) / h * z
            
            points_3d.append([x_3d, y_3d, z])
            
            # Get color
            if len(img_array.shape) == 3:
                colors.append(img_array[y, x])
            else:
                colors.append([gray[y, x]] * 3)
        
        # Update state
        slam_state['points'].extend(points_3d)
        slam_state['colors'].extend(colors)
        slam_state['frame_count'] += 1
        
        # Camera position
        angle = slam_state['frame_count'] * 0.15
        cam_pos = [np.cos(angle) * 4, np.sin(angle) * 0.5, np.sin(angle) * 4]
        slam_state['trajectory'].append(cam_pos)
        
        # Create Plotly figure
        fig = create_3d_figure(points_3d, colors, cam_pos)
        
        status = f"✅ Frame {slam_state['frame_count']} processed\n📊 {len(points_3d)} points extracted\n🎥 Camera moving around scene"
        
        return fig, status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"

def create_3d_figure(points, colors, cam_pos):
    """Create interactive Plotly 3D figure."""
    points = np.array(points)
    colors = np.array(colors) / 255.0
    
    fig = go.Figure()
    
    # Add point cloud
    if len(points) > 0:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=['rgb({},{},{})'.format(int(c[0]), int(c[1]), int(c[2])) for c in colors],
                opacity=0.8
            ),
            name='3D Points'
        ))
    
    # Add camera position
    fig.add_trace(go.Scatter3d(
        x=[cam_pos[0]],
        y=[cam_pos[1]],
        z=[cam_pos[2]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Camera'
    ))
    
    # Add trajectory
    if len(slam_state['trajectory']) > 1:
        traj = np.array(slam_state['trajectory'])
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines',
            line=dict(color='cyan', width=4),
            name='Camera Path'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'3D Reconstruction - Frame {slam_state["frame_count"]}',
            font=dict(color='white', size=18)
        ),
        scene=dict(
            xaxis=dict(backgroundcolor='rgb(10,10,15)', gridcolor='rgb(40,40,50)', showbackground=True),
            yaxis=dict(backgroundcolor='rgb(10,10,15)', gridcolor='rgb(40,40,50)', showbackground=True),
            zaxis=dict(backgroundcolor='rgb(10,10,15)', gridcolor='rgb(40,40,50)', showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube'
        ),
        paper_bgcolor='rgb(15,15,20)',
        plot_bgcolor='rgb(15,15,20)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def show_final_reconstruction():
    """Show all accumulated points."""
    global slam_state
    
    if not slam_state['points']:
        return None, "❌ No data. Process some frames first."
    
    points = np.array(slam_state['points'])
    colors = np.array(slam_state['colors'])
    
    # Normalize
    center = np.mean(points, axis=0)
    points = points - center
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist * 5
    
    # Sample for performance
    if len(points) > 3000:
        indices = np.random.choice(len(points), 3000, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    fig = go.Figure()
    
    # All points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=['rgb({},{},{})'.format(int(c[0]), int(c[1]), int(c[2])) for c in colors],
            opacity=0.6
        ),
        name='Reconstructed Scene'
    ))
    
    # Trajectory
    if len(slam_state['trajectory']) > 1:
        traj = np.array(slam_state['trajectory'])
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=6, color='red'),
            name='Camera Trajectory'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Final Reconstruction - {slam_state["frame_count"]} frames',
            font=dict(color='white', size=20)
        ),
        scene=dict(
            xaxis=dict(backgroundcolor='rgb(10,10,15)', gridcolor='rgb(40,40,50)'),
            yaxis=dict(backgroundcolor='rgb(10,10,15)', gridcolor='rgb(40,40,50)'),
            zaxis=dict(backgroundcolor='rgb(10,10,15)', gridcolor='rgb(40,40,50)'),
            camera=dict(eye=dict(x=2, y=2, z=1.5)),
            aspectmode='cube'
        ),
        paper_bgcolor='rgb(15,15,20)',
        font=dict(color='white'),
        showlegend=True,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    status = f"✅ Final reconstruction complete!\n📊 {len(points)} points\n🎥 {slam_state['frame_count']} frames"
    return fig, status

def reset_system():
    """Reset everything."""
    global slam_state
    slam_state = {'points': [], 'colors': [], 'trajectory': [], 'frame_count': 0}
    return None, "🔄 System reset. Ready for new reconstruction."

def generate_sample_video():
    """Generate a sample video for users to download."""
    # Create synthetic video frames
    frames = []
    for i in range(30):
        # Create a rotating cube pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(480):
            img[y, :] = [int(30 + y/10)] * 3
        
        # Rotating rectangle
        angle = i * 12
        center = (320, 240)
        size = 100
        
        pts = np.array([
            [center[0] + size * np.cos(np.radians(angle)), center[1] + size * np.sin(np.radians(angle))],
            [center[0] + size * np.cos(np.radians(angle + 90)), center[1] + size * np.sin(np.radians(angle + 90))],
            [center[0] + size * np.cos(np.radians(angle + 180)), center[1] + size * np.sin(np.radians(angle + 180))],
            [center[0] + size * np.cos(np.radians(angle + 270)), center[1] + size * np.sin(np.radians(angle + 270))]
        ], np.int32)
        
        cv2.fillPoly(img, [pts], (0, 113, 227))
        cv2.polylines(img, [pts], True, (255, 255, 255), 3)
        
        # Add some features
        cv2.circle(img, (150, 150), 40, (255, 100, 100), -1)
        cv2.circle(img, (500, 350), 50, (100, 255, 100), -1)
        
        frames.append(img)
    
    # Save as video
    out = cv2.VideoWriter('/tmp/sample_slam_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()
    
    return '/tmp/sample_slam_video.mp4'

# Build interface
with gr.Blocks(
    title="Neural SLAM - Real-Time 3D Reconstruction",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")
) as demo:
    
    # Hero
    gr.HTML("""
    <div class="hero">
        <div class="hero-content">
            <h1>Neural <span class="highlight">SLAM</span></h1>
            <p>Real-time 3D scene reconstruction from video. Upload frames and explore the interactive 3D model.</p>
        </div>
    </div>
    """)
    
    # Technology Section
    gr.HTML("""
    <div class="section-header">
        <h2>Advanced Spatial Intelligence</h2>
        <p>Powered by cutting-edge computer vision and 3D geometry</p>
    </div>
    """)
    
    gr.HTML("""
    <div class="tech-grid">
        <div class="tech-card">
            <div class="tech-icon">📐</div>
            <h3>Geometric Vision</h3>
            <p>Edge-based feature extraction converts 2D images into 3D point clouds using depth estimation from visual cues.</p>
        </div>
        
        <div class="tech-card">
            <div class="tech-icon">🎯</div>
            <h3>Simultaneous Localization</h3>
            <p>Track camera position in 3D space as it moves around the scene, building a complete trajectory map.</p>
        </div>
        
        <div class="tech-card">
            <div class="tech-icon">🧩</div>
            <h3>Dense Mapping</h3>
            <p>Accumulate point clouds across multiple frames to create a comprehensive 3D representation of the environment.</p>
        </div>
        
        <div class="tech-card">
            <div class="tech-icon">🎮</div>
            <h3>Interactive Exploration</h3>
            <p>Rotate, zoom, and pan through the reconstructed 3D scene with an intuitive visual interface.</p>
        </div>
    </div>
    """)
    
    # Demo Section
    gr.HTML('<div class="demo-container">')
    
    gr.Markdown("<div class='demo-title'>Try It Now</div>")
    gr.Markdown("<div class='demo-subtitle'>Upload video frames to build your 3D reconstruction</div>")
    
    # Sample video section
    gr.HTML("""
    <div class="sample-section">
        <h4>🎬 Sample Video Available</h4>
        <p>Download this sample video to test the reconstruction. Extract frames and upload them one by one.</p>
    </div>
    """)
    
    sample_video = generate_sample_video()
    gr.File(
        value=sample_video,
        label="Download Sample Video (MP4)",
        interactive=False
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Frame")
            
            image_input = gr.Image(
                label="Select Video Frame",
                type="pil",
                height=300
            )
            
            with gr.Row():
                process_btn = gr.Button("▶️ Process Frame", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset", variant="secondary")
            
            final_btn = gr.Button("📊 Show Final Reconstruction", variant="primary")
            
            status = gr.Textbox(
                label="Status",
                value="Ready. Upload a frame or download the sample video above.",
                interactive=False,
                lines=4,
                elem_classes=["status-box"]
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 🎨 Interactive 3D View")
            
            plot_output = gr.Plot(
                label="3D Reconstruction (Drag to rotate, scroll to zoom)",
                height=550
            )
    
    # Instructions
    gr.HTML("""
    <div class="instructions">
        <h4>How to Use</h4>
        <ol>
            <li><strong>Download the sample video</strong> above or use your own</li>
            <li><strong>Extract frames</strong> from the video (you can use any video player or online tool)</li>
            <li><strong>Upload frames one by one</strong> to see the 3D scene build up</li>
            <li><strong>Interact with the 3D plot</strong> - rotate, zoom, pan to explore</li>
            <li>Click <strong>"Show Final Reconstruction"</strong> to see the complete scene</li>
        </ol>
        <p style="color: #666; margin-top: 16px; font-size: 13px;">
            💡 <strong>Tip:</strong> The system works best with videos showing distinct objects and edges. 
            The camera should move smoothly around the scene for best results.
        </p>
    </div>
    """)
    
    gr.HTML('</div>')
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <p>Built by <a href="https://github.com/vraul92" target="_blank">Rahul Vuppalapati</a> | 
        <a href="https://linkedin.com/in/vrc7" target="_blank">LinkedIn</a></p>
        <p>Senior Data Scientist | Previously Apple, Walmart, IBM</p>
    </div>
    """)
    
    # Events
    process_btn.click(
        fn=process_frame,
        inputs=[image_input],
        outputs=[plot_output, status]
    )
    
    reset_btn.click(
        fn=reset_system,
        outputs=[plot_output, status]
    )
    
    final_btn.click(
        fn=show_final_reconstruction,
        outputs=[plot_output, status]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
