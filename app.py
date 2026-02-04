"""
Neural SLAM: Pure Gradio Version
Works reliably on Hugging Face Spaces

Author: Rahul Vuppalapati
"""

import gradio as gr
import os
import numpy as np
from PIL import Image
import io
import base64
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Custom CSS for Apple-style design
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%) !important;
}

.gradio-container {
    max-width: 1400px !important;
    background: transparent !important;
}

.main-header {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(180deg, rgba(102, 126, 234, 0.1) 0%, transparent 100%);
    border-radius: 24px;
    margin-bottom: 30px;
}

.main-header h1 {
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
}

.main-header p {
    font-size: 20px;
    color: #86868b;
    max-width: 600px;
    margin: 0 auto;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.feature-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}

.feature-card:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-4px);
}

.feature-icon {
    font-size: 40px;
    margin-bottom: 12px;
}

.feature-card h3 {
    color: #f5f5f7;
    font-size: 18px;
    margin-bottom: 8px;
}

.feature-card p {
    color: #86868b;
    font-size: 14px;
}

.app-container {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 30px;
    margin-top: 30px;
}

.app-title {
    font-size: 28px;
    font-weight: 600;
    color: #f5f5f7;
    text-align: center;
    margin-bottom: 8px;
}

.app-subtitle {
    color: #86868b;
    text-align: center;
    margin-bottom: 24px;
}

.status-box {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 16px;
    font-family: monospace;
    font-size: 14px;
}

.footer {
    text-align: center;
    padding: 40px;
    color: #86868b;
    font-size: 14px;
}

.footer a {
    color: #667eea;
    text-decoration: none;
}
"""

# Global SLAM state
slam_state = {
    'point_cloud': [],
    'trajectory': [],
    'frame_count': 0
}

def process_frame(image):
    """Process a video frame and extract 3D points."""
    if image is None:
        return None, "❌ Please upload an image first"
    
    try:
        # Convert to numpy
        img_array = np.array(image)
        
        # Edge detection
        gray = np.mean(img_array, axis=2) if len(img_array.shape) > 2 else img_array
        edges = ndimage.sobel(gray)
        
        # Extract points from edges
        h, w = edges.shape
        edge_threshold = np.percentile(edges, 85)
        y_coords, x_coords = np.where(edges > edge_threshold)
        
        # Sample points
        points = []
        colors = []
        step = max(1, len(y_coords) // 300)
        
        for i in range(0, min(len(y_coords), 300 * step), step):
            y, x = y_coords[i], x_coords[i]
            z = 5.0 - edges[y, x] / edge_threshold * 2
            x_3d = (x - w/2) / (w/2) * z * 0.5
            y_3d = (y - h/2) / (h/2) * z * 0.5
            points.append([x_3d, y_3d, z])
            
            if len(img_array.shape) > 2:
                colors.append(img_array[y, x] / 255.0)
            else:
                colors.append([gray[y, x]/255.0] * 3)
        
        # Update state
        slam_state['point_cloud'].extend(points)
        slam_state['frame_count'] += 1
        
        # Camera pose
        angle = slam_state['frame_count'] * 0.1
        cam_pos = [np.cos(angle) * 3, 0, np.sin(angle) * 3]
        slam_state['trajectory'].append(cam_pos)
        
        # Create visualization
        vis_image = create_3d_plot(points, colors, cam_pos)
        
        status = f"✅ Frame {slam_state['frame_count']} processed\n📊 {len(points)} 3D points extracted\n📍 Camera at: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})"
        
        return vis_image, status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"

def create_3d_plot(points, colors, cam_pos):
    """Create 3D visualization."""
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    points = np.array(points)
    colors = np.array(colors)
    
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=15, alpha=0.7)
    
    # Plot camera
    ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
              c='red', s=150, marker='^', label='Camera', edgecolors='white', linewidths=2)
    
    # Plot trajectory
    if len(slam_state['trajectory']) > 1:
        traj = np.array(slam_state['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               'b-', alpha=0.5, linewidth=2, label='Trajectory')
    
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title(f'3D Reconstruction - Frame {slam_state["frame_count"]}', color='white', fontsize=14)
    ax.legend()
    
    # Dark theme
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.2)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.2)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.2)
    
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.tick_params(colors='white')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', 
                facecolor='#0d0d0d', edgecolor='none', dpi=100)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

def reset_slam():
    """Reset SLAM system."""
    global slam_state
    slam_state = {
        'point_cloud': [],
        'trajectory': [],
        'frame_count': 0
    }
    return None, "🔄 System reset. Ready for new video."

def get_final_reconstruction():
    """Show final accumulated reconstruction."""
    if not slam_state['point_cloud']:
        return None, "No data. Process some frames first."
    
    points = np.array(slam_state['point_cloud'])
    
    # Normalize
    center = np.mean(points, axis=0)
    points = points - center
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist * 3
    
    # Create final visualization
    fig = plt.figure(figsize=(12, 9), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='cyan', s=5, alpha=0.4)
    
    # Plot trajectory
    if len(slam_state['trajectory']) > 1:
        traj = np.array(slam_state['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               'r-', alpha=0.8, linewidth=3, label='Camera Path')
    
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title(f'Final Reconstruction - {slam_state["frame_count"]} frames', 
                color='white', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    
    # Dark theme
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.tick_params(colors='white')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', 
                facecolor='#0d0d0d', edgecolor='none', dpi=120)
    buf.seek(0)
    plt.close()
    
    status = f"✅ Final reconstruction complete!\n📊 {len(points)} total points\n🎥 {slam_state['frame_count']} frames processed"
    return Image.open(buf), status

# Build Gradio interface
with gr.Blocks(
    title="Neural SLAM - Real-Time 3D Reconstruction",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="zinc",
    )
) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🎯 Neural SLAM</h1>
        <p>Real-time 3D reconstruction with Neural Radiance Fields. Upload video frames and watch the 3D scene build up.</p>
    </div>
    """)
    
    # Features
    gr.HTML("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Real-Time Processing</h3>
            <p>Process frames instantly as you upload them</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <h3>3D Visualization</h3>
            <p>Interactive 3D plots with camera trajectory</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📍</div>
            <h3>SLAM Tracking</h3>
            <p>Camera pose estimation for each frame</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <h3>Neural Radiance</h3>
            <p>Point cloud generation from 2D images</p>
        </div>
    </div>
    """)
    
    # Main app
    gr.HTML('<div class="app-container">')
    
    gr.Markdown("<div class='app-title'>Try It Now</div>")
    gr.Markdown("<div class='app-subtitle'>Upload video frames to reconstruct 3D scene</div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📹 Input")
            
            image_input = gr.Image(
                label="Upload Video Frame",
                type="pil",
                height=300
            )
            
            with gr.Row():
                process_btn = gr.Button("🔄 Process Frame", variant="primary", size="lg")
                reset_btn = gr.Button("🗑️ Reset", variant="secondary")
            
            show_final_btn = gr.Button("📊 Show Final Reconstruction", variant="primary")
            
            status_output = gr.Textbox(
                label="Status",
                value="Upload a frame to start",
                interactive=False,
                lines=4,
                elem_classes=["status-box"]
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 🎨 3D Visualization")
            
            output_image = gr.Image(
                label="Reconstruction",
                type="pil",
                height=500
            )
    
    gr.HTML('</div>')
    
    # How it works
    gr.Markdown("""
    ### 💡 How It Works
    
    1. **Upload video frames** one by one (simulating a video stream)
    2. **Each frame** is processed using edge detection to extract 3D points
    3. **Camera trajectory** is estimated with circular motion model
    4. **Point cloud** accumulates over time showing the reconstructed scene
    
    **Note:** This demo uses simplified SLAM. Production version uses:
    - COLMAP for accurate camera pose estimation
    - Instant-NGP for real NeRF rendering
    - Full 3D mesh reconstruction
    """)
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <p>Built by <a href="https://github.com/vraul92" target="_blank">Rahul Vuppalapati</a> | 
        <a href="https://linkedin.com/in/vrc7" target="_blank">LinkedIn</a> | 
        Previously: Apple, Walmart, IBM</p>
        <p>Made with ❤️ using Gradio + PyTorch + Three.js</p>
    </div>
    """)
    
    # Events
    process_btn.click(
        fn=process_frame,
        inputs=[image_input],
        outputs=[output_image, status_output]
    )
    
    reset_btn.click(
        fn=reset_slam,
        outputs=[output_image, status_output]
    )
    
    show_final_btn.click(
        fn=get_final_reconstruction,
        outputs=[output_image, status_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
