import cv2
from ultralytics import YOLO

# Load the trained model
model_path = 'C:\\Users\\joaoa\\PycharmProjects\\spotify\\runs\\train-ALL\\weights\\best.pt'  # Update this path
print(f"Loading model from {model_path}")
model = YOLO(model_path)
print("Model loaded successfully")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame, conf=0.69)  # Lower the confidence threshold if needed
    print("Inference results:", results)  # Print results to debug

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label = model.names[int(box.cls[0])]
            conf = box.conf[0]
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame = cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
