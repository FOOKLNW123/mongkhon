import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import io
import base64

# Configure page
st.set_page_config(
    page_title="🎨 AI แปลงภาพเป็นอนิเมะและสเก็ต",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_anime_model():
    """Load AnimeGANv2 model with caching"""
    try:
        model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained='face_paint_512_v2')
        model.eval()
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดล AnimeGANv2 ได้: {str(e)}")
        return None

def sharpen(img: Image.Image) -> Image.Image:
    """Apply sharpening filter to enhance image quality"""
    img_np = np.array(img)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(sharpened)

def to_anime_full(img: Image.Image, model) -> Image.Image:
    """Convert entire image to anime style using AnimeGANv2"""
    if model is None:
        raise Exception("โมเดลไม่พร้อมใช้งาน")
    
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
    
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
    output_img = transforms.ToPILImage()(output_tensor)
    return sharpen(output_img.resize(original_size))

def to_sketch(img: Image.Image) -> Image.Image:
    """Convert image to pencil sketch style"""
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    inv_img = 255 - img_gray
    blur_img = cv2.GaussianBlur(inv_img, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(img_gray, 255 - blur_img, scale=256)
    return Image.fromarray(sketch)

def get_image_download_link(img, filename, text):
    """Generate download link for processed image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def main():
    # Header
    st.title("🎨 ระบบ AI แปลงภาพจริงเป็นภาพอนิเมะและสเก็ต")
    st.markdown("---")
    
    # Load model
    with st.spinner("กำลังโหลดโมเดล AI... กรุณารอสักครู่"):
        anime_model = load_anime_model()
    
    if anime_model is None:
        st.error("ไม่สามารถโหลดโมเดลได้ กรุณาลองใหม่อีกครั้ง")
        st.stop()
    
    st.success("โมเดล AI พร้อมใช้งานแล้ว! 🚀")
    
    # Instructions
    st.markdown("""
    ### 📝 วิธีใช้งาน:
    1. อัปโหลดภาพที่ต้องการแปลง (รองรับ JPG, PNG, JPEG)
    2. เลือกสไตล์ที่ต้องการ: อนิเมะ หรือ สเก็ต
    3. รอการประมวลผลและดาวน์โหลดผลลัพธ์
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 เลือกภาพที่ต้องการแปลง",
        type=['jpg', 'jpeg', 'png'],
        help="อัปโหลดภาพขนาดไม่เกิน 200MB"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            original_image = Image.open(uploaded_file).convert('RGB')
            
            # Display original image info
            st.markdown("### 📷 ภาพต้นฉบับ")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(original_image, caption="ภาพต้นฉบับ", use_column_width=True)
            
            with col2:
                st.markdown(f"""
                **ข้อมูลภาพ:**
                - ขนาด: {original_image.size[0]} x {original_image.size[1]} พิกเซล
                - โหมดสี: {original_image.mode}
                - ชื่อไฟล์: {uploaded_file.name}
                """)
            
            # Processing options
            st.markdown("### ⚙️ เลือกสไตล์การแปลง")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✨ แปลงเป็นภาพอนิเมะ", use_container_width=True):
                    with st.spinner("กำลังแปลงภาพเป็นสไตล์อนิเมะ... โปรดรอสักครู่"):
                        try:
                            anime_result = to_anime_full(original_image, anime_model)
                            st.session_state['anime_result'] = anime_result
                            st.success("แปลงภาพอนิเมะสำเร็จ! 🎉")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการแปลงภาพอนิเมะ: {str(e)}")
            
            with col2:
                if st.button("🖊️ แปลงเป็นภาพสเก็ต", use_container_width=True):
                    with st.spinner("กำลังแปลงภาพเป็นสไตล์สเก็ต... โปรดรอสักครู่"):
                        try:
                            sketch_result = to_sketch(original_image)
                            st.session_state['sketch_result'] = sketch_result
                            st.success("แปลงภาพสเก็ตสำเร็จ! 🎉")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการแปลงภาพสเก็ต: {str(e)}")
            
            # Display results
            if 'anime_result' in st.session_state or 'sketch_result' in st.session_state:
                st.markdown("### 🎯 ผลลัพธ์")
                
                # Create columns for results
                result_cols = []
                if 'anime_result' in st.session_state:
                    result_cols.append(('anime', st.session_state['anime_result'], "ภาพอนิเมะ"))
                if 'sketch_result' in st.session_state:
                    result_cols.append(('sketch', st.session_state['sketch_result'], "ภาพสเก็ต"))
                
                if len(result_cols) == 1:
                    # Single result
                    result_type, result_img, title = result_cols[0]
                    st.image(result_img, caption=title, use_column_width=True)
                    
                    # Download button
                    filename = f"{uploaded_file.name.split('.')[0]}_{result_type}.png"
                    st.markdown(
                        get_image_download_link(result_img, filename, f"📥 ดาวน์โหลด{title}"),
                        unsafe_allow_html=True
                    )
                
                elif len(result_cols) == 2:
                    # Side by side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        result_type, result_img, title = result_cols[0]
                        st.image(result_img, caption=title, use_column_width=True)
                        filename = f"{uploaded_file.name.split('.')[0]}_{result_type}.png"
                        st.markdown(
                            get_image_download_link(result_img, filename, f"📥 ดาวน์โหลด{title}"),
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        result_type, result_img, title = result_cols[1]
                        st.image(result_img, caption=title, use_column_width=True)
                        filename = f"{uploaded_file.name.split('.')[0]}_{result_type}.png"
                        st.markdown(
                            get_image_download_link(result_img, filename, f"📥 ดาวน์โหลด{title}"),
                            unsafe_allow_html=True
                        )
                
                # Clear results button
                if st.button("🗑️ ล้างผลลัพธ์", use_container_width=True):
                    if 'anime_result' in st.session_state:
                        del st.session_state['anime_result']
                    if 'sketch_result' in st.session_state:
                        del st.session_state['sketch_result']
                    st.rerun()
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📌 ข้อมูลเพิ่มเติม:
    - **โมเดล AI**: AnimeGANv2 (face_paint_512_v2)
    - **การประมวลผล**: รองรับภาพขนาดใหญ่ แต่จะปรับขนาดเป็น 512x512 สำหรับการประมวลผล
    - **คุณภาพ**: ภาพอนิเมะจะถูกเพิ่มความคมชัดด้วย sharpening filter
    - **รูปแบบ**: ดาวน์โหลดเป็นไฟล์ PNG
    """)
    
    st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
        <p>🎨 AI Image Style Transfer | Powered by AnimeGANv2 & OpenCV</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
