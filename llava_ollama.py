import requests
import json
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, List, Optional, Union
import time
import os


class LLaVAClient:
    """Client for interacting with LLaVA model via Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize LLaVA client.
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.model = "llava"
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Error encoding image: {e}")
    
    def encode_pil_image(self, pil_image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def analyze_image(self, 
                     image_input: Union[str, Image.Image], 
                     prompt: str = "Describe this image in detail.",
                     stream: bool = False) -> str:
        """
        Analyze image with LLaVA model.
        
        Args:
            image_input: Path to image file or PIL Image object
            prompt: Text prompt for analysis
            stream: Whether to stream response
            
        Returns:
            Analysis result as string
        """
        # Encode image
        if isinstance(image_input, str):
            image_b64 = self.encode_image(image_input)
        elif isinstance(image_input, Image.Image):
            image_b64 = self.encode_pil_image(image_input)
        else:
            raise ValueError("image_input must be file path or PIL Image")
        
        # Prepare payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": stream
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            full_response += data['response']
                        if data.get('done', False):
                            break
                return full_response
            else:
                # Handle non-streaming response
                return response.json().get('response', '')
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def check_server_status(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False


class ImageGenerator:
    """Generate sample images for testing."""
    
    @staticmethod
    def create_sample_chart():
        """Create a sample chart image."""
        # Generate sample data
        np.random.seed(42)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        sales = [15000, 18000, 22000, 19000, 25000, 28000]
        costs = [12000, 14000, 16000, 15000, 18000, 20000]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(months))
        width = 0.35
        
        ax.bar(x - width/2, sales, width, label='Sales', color='skyblue')
        ax.bar(x + width/2, costs, width, label='Costs', color='lightcoral')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Amount ($)')
        ax.set_title('Monthly Sales vs Costs Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save to PIL Image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return Image.open(buffer)
    
    @staticmethod
    def create_sample_diagram():
        """Create a sample technical diagram."""
        img = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw flowchart elements
        boxes = [
            (100, 100, 200, 150, "Input Data"),
            (300, 100, 400, 150, "Processing"),
            (500, 100, 600, 150, "Analysis"),
            (300, 250, 400, 300, "Validation"),
            (500, 250, 600, 300, "Output")
        ]
        
        # Draw boxes and labels
        for x1, y1, x2, y2, label in boxes:
            draw.rectangle([x1, y1, x2, y2], outline='black', fill='lightblue', width=2)
            # Calculate text position (approximate center)
            text_x = x1 + (x2 - x1) // 2 - len(label) * 3
            text_y = y1 + (y2 - y1) // 2 - 10
            draw.text((text_x, text_y), label, fill='black')
        
        # Draw arrows
        arrows = [
            (200, 125, 300, 125),  # Input to Processing
            (400, 125, 500, 125),  # Processing to Analysis
            (350, 150, 350, 250),  # Processing to Validation
            (450, 275, 500, 275),  # Validation to Output
        ]
        
        for x1, y1, x2, y2 in arrows:
            draw.line([x1, y1, x2, y2], fill='black', width=3)
            # Simple arrowhead
            if x2 > x1:  # Right arrow
                draw.polygon([x2-10, y2-5, x2, y2, x2-10, y2+5], fill='black')
            elif y2 > y1:  # Down arrow
                draw.polygon([x2-5, y2-10, x2, y2, x2+5, y2-10], fill='black')
        
        return img
    
    @staticmethod
    def create_sample_photo():
        """Create a sample photo-like image."""
        # Create a landscape scene
        img = Image.new('RGB', (800, 600), 'skyblue')
        draw = ImageDraw.Draw(img)
        
        # Sky gradient (simplified)
        for y in range(0, 300):
            color_intensity = int(135 + (120 * y / 300))
            draw.line([(0, y), (800, y)], fill=(color_intensity, color_intensity + 20, 255))
        
        # Ground
        draw.rectangle([0, 300, 800, 600], fill='green')
        
        # Sun
        draw.ellipse([650, 50, 750, 150], fill='yellow', outline='orange', width=3)
        
        # Mountains
        mountain_points = [(0, 300), (150, 200), (300, 250), (450, 180), (600, 220), (800, 300)]
        draw.polygon(mountain_points, fill='gray', outline='darkgray')
        
        # Trees (simplified)
        tree_positions = [100, 250, 400, 550, 700]
        for x in tree_positions:
            # Trunk
            draw.rectangle([x-5, 280, x+5, 320], fill='brown')
            # Leaves
            draw.ellipse([x-20, 250, x+20, 290], fill='darkgreen')
        
        return img


def run_comprehensive_examples():
    """Run comprehensive examples of LLaVA usage."""
    
    print("🚀 LLaVA Model with Ollama - Comprehensive Examples")
    print("=" * 55)
    
    # Initialize client
    client = LLaVAClient()
    
    # Check server status
    print("📡 Checking Ollama server status...")
    if not client.check_server_status():
        print("❌ Ollama server is not running!")
        print("💡 Please start Ollama server: ollama serve")
        print("💡 And pull LLaVA model: ollama pull llava")
        return
    
    print("✅ Ollama server is running!")
    
    # Create image generator
    img_gen = ImageGenerator()
    
    # Example 1: Analyze a chart
    print("\n📊 Example 1: Analyzing a Business Chart")
    print("-" * 40)
    
    chart_img = img_gen.create_sample_chart()
    chart_img.save("sample_chart.png")
    
    chart_analysis = client.analyze_image(
        chart_img,
        "Analyze this business chart. What trends do you see? What insights can you provide?"
    )
    print("🔍 Analysis:")
    print(chart_analysis)
    
    # Example 2: Analyze a technical diagram
    print("\n🔧 Example 2: Analyzing a Technical Diagram")
    print("-" * 45)
    
    diagram_img = img_gen.create_sample_diagram()
    diagram_img.save("sample_diagram.png")
    
    diagram_analysis = client.analyze_image(
        diagram_img,
        "Describe this flowchart. What process does it represent? Identify the components and flow."
    )
    print("🔍 Analysis:")
    print(diagram_analysis)
    
    # Example 3: Analyze a scene
    print("\n🏞️ Example 3: Analyzing a Landscape Scene")
    print("-" * 42)
    
    scene_img = img_gen.create_sample_photo()
    scene_img.save("sample_scene.png")
    
    scene_analysis = client.analyze_image(
        scene_img,
        "Describe this landscape image. What elements do you see? What's the mood or atmosphere?"
    )
    print("🔍 Analysis:")
    print(scene_analysis)
    
    # Example 4: Comparative analysis
    print("\n⚖️ Example 4: Comparative Analysis")
    print("-" * 35)
    
    comparison_prompt = """
    Compare and contrast these three images I've shown you:
    1. The business chart
    2. The technical diagram  
    3. The landscape scene
    
    What are the key differences in visual style, purpose, and information conveyed?
    """
    
    # For comparison, we'll use the chart again but with a different prompt
    comparison_analysis = client.analyze_image(
        chart_img,
        "This is one of several images. Focus on its visual design elements, color scheme, and data presentation style. How would you categorize this type of visualization?"
    )
    print("🔍 Analysis:")
    print(comparison_analysis)


def create_analysis_dashboard(results: Dict[str, str]):
    """Create a dashboard showing analysis results."""
    
    # Create analysis metrics
    metrics = []
    for name, analysis in results.items():
        word_count = len(analysis.split())
        char_count = len(analysis)
        sentence_count = analysis.count('.') + analysis.count('!') + analysis.count('?')
        
        metrics.append({
            'Image': name,
            'Word Count': word_count,
            'Character Count': char_count,
            'Sentences': sentence_count,
            'Avg Words/Sentence': round(word_count / max(sentence_count, 1), 1)
        })
    
    df = pd.DataFrame(metrics)
    
    # Create dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LLaVA Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Word count comparison
    ax1.bar(df['Image'], df['Word Count'], color='skyblue', alpha=0.7)
    ax1.set_title('Analysis Word Count by Image')
    ax1.set_ylabel('Words')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Character count
    ax2.plot(df['Image'], df['Character Count'], marker='o', color='green', linewidth=2)
    ax2.set_title('Character Count Analysis')
    ax2.set_ylabel('Characters')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sentences vs Avg Words per Sentence
    scatter = ax3.scatter(df['Sentences'], df['Avg Words/Sentence'], 
                         s=df['Word Count']/2, alpha=0.6, color='coral')
    ax3.set_xlabel('Number of Sentences')
    ax3.set_ylabel('Average Words per Sentence')
    ax3.set_title('Analysis Complexity (bubble size = total words)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax4.axis('tight')
    ax4.axis('off')
    table_data = df[['Image', 'Word Count', 'Sentences']].values
    table = ax4.table(cellText=table_data,
                     colLabels=['Image Type', 'Words', 'Sentences'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Analysis Summary Table')
    
    plt.tight_layout()
    plt.savefig('llava_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()


def advanced_image_analysis_example():
    """Advanced example with image preprocessing and detailed analysis."""
    
    print("\n🔬 Advanced Image Analysis Example")
    print("-" * 40)
    
    client = LLaVAClient()
    
    # Create a complex image with multiple elements
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Line chart
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    axes[0, 0].plot(x, y1, label='sin(x)', color='blue')
    axes[0, 0].plot(x, y2, label='cos(x)', color='red')
    axes[0, 0].set_title('Trigonometric Functions')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Subplot 2: Histogram
    data = np.random.normal(100, 15, 1000)
    axes[0, 1].hist(data, bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('Normal Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Subplot 3: Scatter plot
    x_scatter = np.random.randn(100)
    y_scatter = 2 * x_scatter + np.random.randn(100)
    axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6, color='purple')
    axes[1, 0].set_title('Correlation Plot')
    axes[1, 0].set_xlabel('X values')
    axes[1, 0].set_ylabel('Y values')
    
    # Subplot 4: Heatmap
    heatmap_data = np.random.rand(10, 10)
    im = axes[1, 1].imshow(heatmap_data, cmap='viridis')
    axes[1, 1].set_title('Random Heatmap')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save to PIL Image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    complex_img = Image.open(buffer)
    plt.close()
    
    # Analyze the complex image
    complex_analysis = client.analyze_image(
        complex_img,
        """Analyze this complex data visualization dashboard. For each subplot:
        1. Identify the type of chart/plot
        2. Describe what data pattern it shows
        3. Comment on the visual design and clarity
        4. Suggest any improvements
        
        Also provide an overall assessment of the dashboard's effectiveness."""
    )
    
    print("🔍 Complex Analysis Result:")
    print(complex_analysis)
    
    return complex_analysis


if __name__ == "__main__":
    # Run main examples
    run_comprehensive_examples()
    
    # Run advanced example
    advanced_result = advanced_image_analysis_example()
    
    # Create sample results for dashboard
    sample_results = {
        "Business Chart": "This chart shows monthly sales versus costs data with clear trends indicating growth in both metrics. The visualization uses effective color coding with blue bars for sales and coral bars for costs, making it easy to compare the two metrics across six months from January to June.",
        "Technical Diagram": "This flowchart illustrates a data processing workflow with five key components: Input Data, Processing, Analysis, Validation, and Output. The diagram shows a logical flow with decision points and feedback loops, typical of data pipeline architectures.",
        "Landscape Scene": "This landscape image depicts a serene natural scene with mountains, trees, and a bright sun in a blue sky. The composition creates a peaceful atmosphere with good use of color gradients from sky blue to green ground cover.",
        "Complex Dashboard": advanced_result
    }
    
    # Create analysis dashboard
    print("\n📊 Creating Analysis Dashboard...")
    create_analysis_dashboard(sample_results)
    
    print("\n✅ All examples completed successfully!")
    print("📁 Generated files:")
    print("   - sample_chart.png")
    print("   - sample_diagram.png") 
    print("   - sample_scene.png")
    print("   - llava_analysis_dashboard.png")
    
    print("\n💡 Tips for using LLaVA:")
    print("   1. Use specific, detailed prompts for better analysis")
    print("   2. Try different image types (charts, photos, diagrams)")
    print("   3. Experiment with comparative analysis across multiple images")
    print("   4. Use the streaming option for real-time responses")
    print("   5. Preprocess images for better quality if needed")
