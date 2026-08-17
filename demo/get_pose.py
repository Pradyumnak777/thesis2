'''
inspect the WHAM output
'''
import joblib
import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger attach on port 5678...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

BASE = "smpl_info/expert_exo"


if __name__ == "__main__":
    
    wham_output      = joblib.load(f"{BASE}/wham_output.pkl")
    slam_results     = joblib.load(f"{BASE}/slam_results.pth")
    tracking_results = joblib.load(f"{BASE}/tracking_results.pth")

    
    #inspecting the files
    print("\nPKL Type: ", type(wham_output))
    print("\nPTH(slam) Type:", type(slam_results))
    print("\nPTH(tracking) Type:", type(tracking_results))
    
    
    #get the first person pose information (over all frames)
    pose_info = wham_output[0]
    joblib.dump(pose_info, "pose_info.joblib")
    
    