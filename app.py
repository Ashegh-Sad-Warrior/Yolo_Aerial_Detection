# !pip install ultralytics
# !pip install gradio

import cv2
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import os

# بررسی وجود فایل مدل
if os.path.exists('best.pt'):
    print("Model file found.")
else:
    print("Model file not found. Please upload 'best.pt' to the Space.")

# بارگذاری مدل آموزش‌دیده شما
model = YOLO('best.pt')  # مسیر مدل را به صورت نسبی تنظیم کنید

# تعریف نام کلاس‌ها به انگلیسی و فارسی
class_names = {
    0: ('plane', 'هواپیما'),
    1: ('ship', 'کشتی'),
    2: ('storage tank', 'مخزن ذخیره'),
    3: ('baseball diamond', 'زمین بیسبال'),
    4: ('tennis court', 'زمین تنیس'),
    5: ('basketball court', 'زمین بسکتبال'),
    6: ('ground track field', 'زمین دو و میدانی'),
    7: ('harbor', 'بندرگاه'),
    8: ('bridge', 'پل'),
    9: ('large vehicle', 'خودرو بزرگ'),
    10: ('small vehicle', 'خودرو کوچک'),
    11: ('helicopter', 'هلیکوپتر'),
    12: ('roundabout', 'میدان'),
    13: ('soccer ball field', 'زمین فوتبال'),
    14: ('swimming pool', 'استخر شنا')
}

# رنگ‌ها برای هر کلاس (BGR برای OpenCV)
colors = {
    0: (255, 0, 0),       # قرمز
    1: (0, 255, 0),       # سبز
    2: (0, 0, 255),       # آبی
    3: (255, 255, 0),     # زرد
    4: (255, 0, 255),     # مجنتا
    5: (0, 255, 255),     # فیروزه‌ای
    6: (128, 0, 128),     # بنفش
    7: (255, 165, 0),     # نارنجی
    8: (0, 128, 0),       # سبز تیره
    9: (128, 128, 0),     # زیتونی
    10: (0, 255, 0),      # سبز روشن برای class_id=10
    11: (0, 128, 128),    # سبز نفتی
    12: (0, 0, 128),      # نیوی
    13: (75, 0, 130),     # ایندیگو
    14: (199, 21, 133)    # رز متوسط
}

# تابع برای تشخیص اشیاء در تصاویر
def detect_and_draw_image(input_image):
    try:
        # تبدیل تصویر PIL به آرایه NumPy (RGB)
        input_image_np = np.array(input_image)
        print("Image converted to NumPy array.")

        # اجرای مدل روی تصویر با استفاده از آرایه NumPy (RGB)
        results = model.predict(source=input_image_np, conf=0.3)
        print("Model prediction completed.")

        # دسترسی به نتایج OBB
        if hasattr(results[0], 'obb') and results[0].obb is not None:
            obb_results = results[0].obb
            print("Accessed obb_results.")
        else:
            print("No 'obb' attribute found in results[0].")
            obb_results = None

        # بررسی وجود جعبه‌های شناسایی شده
        if obb_results is None or len(obb_results.data) == 0:
            print("هیچ شیء شناسایی نشده است.")
            df = pd.DataFrame({
                'Label (English)': [],
                'Label (Persian)': [],
                'Object Count': []
            })
            return input_image, df

        counts = {}
        # پردازش نتایج و رسم جعبه‌ها
        for obb, conf, cls in zip(obb_results.data.cpu().numpy(), obb_results.conf.cpu().numpy(), obb_results.cls.cpu().numpy()):
            x_center, y_center, width, height, rotation = obb[:5]
            class_id = int(cls)
            confidence = float(conf)

            # رسم جعبه چرخان با استفاده از OpenCV
            rect = ((x_center, y_center), (width, height), rotation * 180.0 / np.pi)  # تبدیل رادیان به درجه
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            color = colors.get(class_id, (0, 255, 0))
            cv2.drawContours(input_image_np, [box_points], 0, color, 2)
            print(f"Drawn OBB for class_id {class_id} with confidence {confidence}.")

            # رسم برچسب
            label_en, label_fa = class_names.get(class_id, ('unknown', 'ناشناخته'))
            cv2.putText(input_image_np, f'{label_en}: {confidence:.2f}',
                        (int(x_center), int(y_center)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # شمارش اشیاء
            counts[label_en] = counts.get(label_en, 0) + 1

        # تبدیل تصویر به RGB برای Gradio
        image_rgb = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(image_rgb)
        print("Image converted back to RGB for Gradio.")

        # ایجاد DataFrame برای نمایش نتایج
        df = pd.DataFrame({
            'Label (English)': list(counts.keys()),
            'Label (Persian)': [class_names.get(k, ('unknown', 'ناشناخته'))[1] for k in counts.keys()],
            'Object Count': list(counts.values())
        })
        print("DataFrame created.")

        return output_image, df

    except Exception as e:
        print(f"Error in detect_and_draw_image: {e}")
        df = pd.DataFrame({
            'Label (English)': [],
            'Label (Persian)': [],
            'Object Count': []
        })
        return input_image, df

# تابع برای تشخیص اشیاء در ویدئوها
def detect_and_draw_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        overall_counts = {}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count +=1
            print(f"Processing frame {frame_count}")

            # تغییر اندازه فریم
            frame = cv2.resize(frame, (640, 480))

            # اجرای مدل روی فریم
            results = model.predict(source=frame, conf=0.3)
            print(f"Model prediction completed for frame {frame_count}.")

            # دسترسی به نتایج OBB
            if hasattr(results[0], 'obb') and results[0].obb is not None:
                obb_results = results[0].obb
                print("Accessed obb_results for frame.")
            else:
                print("No 'obb' attribute found in results[0] for frame.")
                obb_results = None

            if obb_results is not None and len(obb_results.data) > 0:
                for obb, conf, cls in zip(obb_results.data.cpu().numpy(), obb_results.conf.cpu().numpy(), obb_results.cls.cpu().numpy()):
                    x_center, y_center, width, height, rotation = obb[:5]
                    class_id = int(cls)
                    confidence = float(conf)

                    # رسم جعبه چرخان با استفاده از OpenCV
                    rect = ((x_center, y_center), (width, height), rotation * 180.0 / np.pi)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int0(box_points)
                    color = colors.get(class_id, (0, 255, 0))
                    cv2.drawContours(frame, [box_points], 0, color, 2)
                    print(f"Drawn OBB for class_id {class_id} with confidence {confidence} in frame {frame_count}.")

                    # رسم برچسب
                    label_en, label_fa = class_names.get(class_id, ('unknown', 'ناشناخته'))
                    cv2.putText(frame, f"{label_en}: {confidence:.2f}",
                                (int(x_center), int(y_center)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # شمارش اشیاء
                    overall_counts[label_en] = overall_counts.get(label_en, 0) + 1

            frames.append(frame)
            print(f"Frame {frame_count} processed.")

        cap.release()
        print("Video processing completed.")

        # ذخیره ویدئو پردازش‌شده در یک فایل موقت
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            output_path = tmpfile.name
        print(f"Saving processed video to {output_path}")

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
        for idx, frame in enumerate(frames):
            out.write(frame)
            if idx % 100 == 0:
                print(f"Writing frame {idx} to video.")

        out.release()
        print("Video saved.")

        # ایجاد DataFrame برای ذخیره نتایج
        df = pd.DataFrame({
            'Label (English)': list(overall_counts.keys()),
            'Label (Persian)': [class_names.get(k, ('unknown', 'ناشناخته'))[1] for k in overall_counts.keys()],
            'Object Count': list(overall_counts.values())
        })
        print("DataFrame created.")

        return output_path, df

    except Exception as e:
        print(f"Error in detect_and_draw_video: {e}")
        # در صورت بروز خطا، بازگرداندن ویدئوی اصلی بدون تغییر و یک DataFrame خالی
        return video_path, pd.DataFrame({
            'Label (English)': [],
            'Label (Persian)': [],
            'Object Count': []
        })

# رابط کاربری Gradio برای تصاویر
image_interface = gr.Interface(
    fn=detect_and_draw_image,
    inputs=gr.Image(type="pil", label="بارگذاری تصویر"),
    outputs=[gr.Image(type="pil", label="تصویر پردازش شده"), gr.Dataframe(label="تعداد اشیاء")],
    title="تشخیص اشیاء در تصاویر هوایی",
    description="یک تصویر هوایی بارگذاری کنید تا اشیاء شناسایی شده و تعداد آن‌ها را ببینید.",
    examples=[
        'Examples/images/areial_car.jpg',
        'Examples/images/arieal_car_1.jpg',
        'Examples/images/t.jpg'
    ]
)

# رابط کاربری Gradio برای ویدئوها
video_interface = gr.Interface(
    fn=detect_and_draw_video,
    inputs=gr.Video(label="بارگذاری ویدئو"),
    outputs=[gr.Video(label="ویدئوی پردازش شده"), gr.Dataframe(label="تعداد اشیاء")],
    title="تشخیص اشیاء در ویدئوها",
    description="یک ویدئو بارگذاری کنید تا اشیاء شناسایی شده و تعداد آن‌ها را ببینید.",
    examples=[
        'Examples/video/city.mp4',
        'Examples/video/airplane.mp4'
    ]
)

# اجرای برنامه با استفاده از رابط کاربری تب‌دار
app = gr.TabbedInterface([image_interface, video_interface], ["تشخیص تصویر", "تشخیص ویدئو"])
app.launch()
