import cv2
import os


def capture_album_covers(output_dir='dataset/images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    album_name = input("Enter the album name: ").replace(" ", "_").replace("/", "_")

    # Initialize the image counter based on existing images in the directory
    existing_files = os.listdir(output_dir)
    img_counter = len([name for name in existing_files if name.startswith(f'{album_name}_') and name.endswith('.jpg')])

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    print(f"Saving images to directory: {output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow('Capture Album Covers - Press SPACE to capture, ESC to exit', frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = os.path.join(output_dir, f"{album_name}_{img_counter}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_album_covers()
