import cv2
import mediapipe as mp

# Define indices for two landmarks to calculate distance
index1 = 4
index2 = 8

# Open the video capture
vid = cv2.VideoCapture(0)

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Main function to process video feed
def main():
    while True:
        # Read a frame from the video feed
        ret, frame = vid.read()
        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame using Mediapipe Hands
        result = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if result.multi_hand_landmarks:
            # Get landmarks for the first hand
            landmarks = result.multi_hand_landmarks[0].landmark

            # Calculate pixel coordinates for two specified landmarks
            point1 = (int(landmarks[index1].x * frame.shape[1]), int(landmarks[index1].y * frame.shape[0]))
            point2 = (int(landmarks[index2].x * frame.shape[1]), int(landmarks[index2].y * frame.shape[0]))

            # Calculate distance between the two landmarks
            dist = calculate_dist(point1, point2)
            # Normalize distance and map it to brightness range
            dist = dist / 300 * 100
            brightness = linear_interpolation(int(dist), 10, 125, 960, 96000)

            # Print brightness value
            print(brightness)

            # Write brightness value to the backlight file
            b_file = open("/sys/class/backlight/intel_backlight/brightness", 'w')
            b_file.write(brightness)
            b_file.close()

            # Draw hand landmarks and connections on the frame
            for landmark in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('frame', frame)

        # Check for 'q' key to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close all windows
    vid.release()
    cv2.destroyAllWindows()

# Function to calculate Euclidean distance between two points
def calculate_dist(point1, point2):
    return (((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)) ** (1/2)

# Function for linear interpolation
def linear_interpolation(value, start, end, start_value, end_value):
    if value <= 10:
        return '960'
    elif value >= 120:
        return '96000'
    return str(int(start_value + (end_value - start_value) * ((value - start) / (end - start))))

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
