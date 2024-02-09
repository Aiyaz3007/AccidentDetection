def isColab():
    try:
        import google.colab 
        return True
    except ModuleNotFoundError:
        return False
    
def collate_fn(batch):
    return tuple(zip(*batch))


def save_video(frame_list:list,dst:str):
    try:
        if not frame_list:
            print("Error: Empty frame list.")
            return

        # Get the height and width of the frames from the first frame
        height, width, _ = frame_list[0].shape
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dst, fourcc, 25.0, (width, height))

        if not out.isOpened():
            print("Error: Unable to open the output video file.")
            return

        # Write frames to the video
        for frame in frame_list:
            out.write(frame)

        # Release the VideoWriter
        out.release()
        print("File_Saved!")
    except Exception:
        print("Error occured while saving!")