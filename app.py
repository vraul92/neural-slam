"""
Neural SLAM: Real-Time 3D Reconstruction
Flask + Gradio application with Apple-style landing page

Author: Rahul Vuppalapati
GitHub: https://github.com/vraul92
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, send_from_directory
import gradio as gr
from PIL import Image
import io
import base64

# Create Flask app
app = Flask(__name__)

# Global SLAM state
SLAM_STATE = {
    'initialized': False,
    'frames': [],
    'point_cloud': None,
    'camera_poses': []
}


class SimpleNeuralSLAM:
    """Simplified Neural SLAM for demo purposes."""
    
    def __init__(self):
        self.frames = []
        self.point_cloud = []
        self.camera_trajectory = []
        
    def process_frame(self, image: Image.Image, frame_num: int) -> dict:
        """Process a single video frame."""
        # Convert to numpy
        img_array = np.array(image)
        
        # Extract features (simplified - edge detection)
        from scipy import ndimage
        gray = np.mean(img_array, axis=2) if len(img_array.shape) > 2 else img_array
        edges = ndimage.sobel(gray)
        
        # Generate 3D points (simplified projection)
        h, w = edges.shape
        points = []
        colors = []
        
        # Sample points from edges
        edge_threshold = np.percentile(edges, 90)
        y_coords, x_coords = np.where(edges > edge_threshold)
        
        # Subsample
        step = max(1, len(y_coords) // 500)
        for i in range(0, len(y_coords), step):
            if len(points) >= 500:
                break
            
            y, x = y_coords[i], x_coords[i]
            
            # Project to 3D (simplified pinhole camera)
            z = 5.0 - edges[y, x] / edge_threshold * 2  # Depth from edge strength
            x_3d = (x - w/2) / (w/2) * z * 0.5
            y_3d = (y - h/2) / (h/2) * z * 0.5
            
            points.append([x_3d, y_3d, z])
            
            # Get color from original image
            if len(img_array.shape) > 2:
                colors.append(img_array[y, x] / 255.0)
            else:
                colors.append([gray[y, x]/255.0] * 3)
        
        # Camera pose (simplified circular motion)
        angle = frame_num * 0.1
        camera_pose = {
            'position': [np.cos(angle) * 3, 0, np.sin(angle) * 3],
            'rotation': angle
        }
        
        self.point_cloud.extend(points)
        self.camera_trajectory.append(camera_pose['position'])
        
        return {
            'points': np.array(points),
            'colors': np.array(colors),
            'camera_pose': camera_pose,
            'frame_num': frame_num
        }
    
    def get_reconstruction_data(self) -> dict:
        """Get complete reconstruction data for visualization."""
        if not self.point_cloud:
            return None
        
        points = np.array(self.point_cloud)
        
        # Normalize
        center = np.mean(points, axis=0)
        points = points - center
        
        # Scale
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist * 2
        
        return {
            'points': points.tolist(),
            'trajectory': self.camera_trajectory,
            'num_frames': len(self.frames)
        }


# Global SLAM instance
slam_system = SimpleNeuralSLAM()


def process_video_frame(image, frame_num):
    """Process a video frame and return 3D visualization."""
    try:
        if image is None:
            return None, "No image provided"
        
        # Process frame
        result = slam_system.process_frame(image, frame_num)
        
        # Create visualization
        vis_image = create_3d_visualization(result)
        
        status = f"✅ Processed frame {frame_num + 1}\n📊 Extracted {len(result['points'])} 3D points"
        
        return vis_image, status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


def create_3d_visualization(result: dict) -> Image.Image:
    """Create a 2D visualization of 3D points."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    points = result['points']
    colors = result['colors']
    
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=10, alpha=0.6)
    
    # Plot camera
    cam_pos = result['camera_pose']['position']
    ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], 
              c='red', s=100, marker='^', label='Camera')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction - Frame {result["frame_num"] + 1}')
    ax.legend()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def reset_slam():
    """Reset SLAM system."""
    global slam_system
    slam_system = SimpleNeuralSLAM()
    return None, "🔄 SLAM system reset. Ready for new video."


# Gradio Interface
def create_gradio_interface():
    """Create Gradio interface for the demo."""
    
    with gr.Blocks(
        title="Neural SLAM Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        )
    ) as demo:
        
        gr.Markdown("""
        # 🎯 Neural SLAM Demo
        ### Real-time 3D reconstruction from video
        
        Upload video frames to see 3D reconstruction in action.
        """)
        
        frame_num = gr.State(0)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Input")
                
                image_input = gr.Image(
                    label="Upload Video Frame",
                    type="pil"
                )
                
                with gr.Row():
                    process_btn = gr.Button("🔄 Process Frame", variant="primary")
                    reset_btn = gr.Button("🗑️ Reset", variant="secondary")
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Upload a frame to start",
                    interactive=False,
                    lines=3
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎨 3D Visualization")
                
                output_image = gr.Image(
                    label="Reconstruction",
                    type="pil"
                )
        
        # Event handlers
        process_btn.click(
            fn=lambda img, fn: process_video_frame(img, fn),
            inputs=[image_input, frame_num],
            outputs=[output_image, status_text]
        ).then(
            fn=lambda x: x + 1,
            inputs=[frame_num],
            outputs=[frame_num]
        )
        
        reset_btn.click(
            fn=reset_slam,
            outputs=[output_image, status_text]
        ).then(
            fn=lambda: 0,
            outputs=[frame_num]
        )
        
        gr.Markdown("""
        ---
        
        ### 💡 How it works:
        
        1. **Upload video frames** one by one (simulating video stream)
        2. **Each frame** is processed to extract 3D points
        3. **Camera pose** is estimated for each frame
        4. **Point cloud** accumulates over time
        
        **Note:** This is a simplified demo. Production version uses:
        - instant-ngp for real NeRF
        - COLMAP for accurate SLAM
        - Full 3D viewer with Three.js
        """)
    
    return demo


# Flask Routes
@app.route('/')
def index():
    """Render landing page."""
    return render_template('index.html')


@app.route('/static/<path:path>')
def static_files(path):
    """Serve static files."""
    return send_from_directory('static', path)


@app.route('/gradio')
def gradio_app():
    """Serve Gradio app in iframe."""
    from gradio.routes import App
    gradio_interface = create_gradio_interface()
    return gradio_interface.launch(
        inline=True,
        prevent_thread_lock=True,
        show_error=True
    )


# Main entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    
    # For local development
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # Production - mount Gradio at /gradio
        gradio_interface = create_gradio_interface()
        
        # Mount Gradio app
        from werkzeug.middleware.dispatcher import DispatcherMiddleware
        from werkzeug.serving import run_simple
        
        application = DispatcherMiddleware(
            app,
            {'/gradio': gradio_interface.app}
        )
        
        print(f"🚀 Starting Neural SLAM on port {port}")
        run_simple('0.0.0.0', port, application, use_reloader=False)
