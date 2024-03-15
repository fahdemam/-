import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import requests
from io import BytesIO

# Load the pre-trained model
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Load the SRGAN model
srgan = tf.keras.models.load_model("srgan_model.h5")

# تعريف الدالة لرسم الصورة
def draw_image():
    # استخراج عنوان الصورة من مربع النص
    image_url = image_entry.get()

    # Load the image from the specified URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Load and preprocess the image for VGG16 model
    processed_image = preprocess_image_vgg(image)

    # Predict the image features using VGG16
    features = model.predict(processed_image)

    # تحسين جودة الصورة باستخدام نموذج SRGAN
    improved_image = srgan_predict(image)

    # Save the enhanced image
    enhanced_image_path = 'enhanced_image.jpg'
    enhanced_image.save(enhanced_image_path)

    # عرض الصورة المحسنة
    tk_image = ImageTk.PhotoImage(improved_image)
    canvas.create_image(200, 200, image=tk_image)
    canvas.image = tk_image

    # عرض رسالة تأكيد
    messagebox.showinfo("تم", "تم تحسين جودة الصورة وحفظها كملف جديد")

# تعريف الدالة لتحسين جودة الصورة باستخدام نموذج SRGAN
def srgan_predict(image):
    # تحسين الصورة باستخدام نموذج SRGAN
    improved_image = srgan.predict(np.expand_dims(image, axis=0))
    improved_image = (improved_image[0] * 255).astype(np.uint8)
    return Image.fromarray(improved_image)

# تعريف الدالة لتحضير الصورة لنموذج VGG16
def preprocess_image_vgg(image):
    # Resize the image to (224, 224)
    image_resized = image.resize((224, 224))
    # Convert the image to array
    image_array = np.array(image_resized)
    # Expand dimensions to match VGG16 input shape
    image_array = np.expand_dims(image_array, axis=0)
    # Preprocess the image for VGG16
    processed_image = tf.keras.applications.vgg16.preprocess_input(image_array)
    return processed_image

# إنشاء نافذة
root = tk.Tk()
root.title("تطبيق تحسين جودة الصور")

# إضافة مربع نص لإدخال عنوان الصورة
image_label = tk.Label(root, text="عنوان الصورة:")
image_label.pack()
image_entry = tk.Entry(root)
image_entry.pack()

# إنشاء زر لرسم الصورة
draw_button = tk.Button(root, text="تحسين الصورة", command=draw_image)
draw_button.pack()

# إنشاء قماش لعرض الصورة
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

# تشغيل البرنامج
root.mainloop()
