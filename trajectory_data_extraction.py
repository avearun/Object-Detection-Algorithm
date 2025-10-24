from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2
import sys

def select_video_file():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Video File",
        "",
        "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
    )
    
    return file_path if file_path else None


if __name__ == "__main__":
    path = select_video_file()
    if path:
        print("Selected:", path)
    else:
        print("No file selected.")

# Process Video
    if path:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Here you can add code to process each frame
            cv2.imshow('Video Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()