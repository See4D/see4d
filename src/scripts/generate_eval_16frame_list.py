import json
import sys
import os

def check_context(vid_data, eval_data):
    for k, vid in vid_data.items():
        if vid is None:
            assert eval_data[k] is None
            continue

        assert vid['context'] == eval_data[k]['context']

def sample_m_elements_uniformly(the_list, m):
    """
    Uniformly samples m elements from the_list, including the first and last elements.
    
    Parameters:
    - the_list: A list containing N elements.
    - m: The number of elements to sample from the_list.
    
    Returns:
    - A list containing m uniformly sampled elements from the_list.
    """
    N = len(the_list)
    if m >= N:
        return the_list
    step = (N - 1) / (m - 1)
    sampled_elements = [the_list[int(i * step)] for i in range(m)]
    return sampled_elements


if __name__ == '__main__':
    eval_path = sys.argv[1]
    eval_data = json.load(open(eval_path))
    
    vid_path = eval_path[:-5] + "_video.json"
    if os.path.isfile(vid_path):
        vid_data = json.load(open(vid_path))
    else:
        vid_data = {}
        for k, e in eval_data.items():
            vid_data[k] = e
            if e is not None:
                c1, c2 = vid_data[k]['context']
                vid_data[k]['target'] = [ci for ci in range(c1,c2+1)]

                assert vid_data[k]['target'][0] == vid_data[k]['context'][0] and vid_data[k]['target'][-1] == vid_data[k]['context'][-1]
                
        with open(vid_path, "w") as f:
            json.dump(vid_data, f, indent=4)
        
    num_frames = 16 # 
    
    save_path = eval_path[:-5] + "_16frames.json"
    
    save_dict = {}

    for k, vid in vid_data.items():
        save_dict[k] = vid
        if vid is None:
            assert eval_data[k] is None
        else:
            assert vid['context'] == eval_data[k]['context']
            ## NOTE Vid['target'] already contain the context frames
            target_list = vid['target']

            if len(target_list) >= num_frames:
                target_frame_list = sample_m_elements_uniformly(target_list, num_frames)
                assert len(target_frame_list) == num_frames and target_list[0] == vid['context'][0] and target_list[-1] == vid['context'][-1]
                save_dict[k]['target'] = target_frame_list
            
            else:
                # only evaluate more than 16 frames
                save_dict[k] = None
                print(f"{k} has less than {num_frames} frames")
    
    # print(save_dict)
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=4)