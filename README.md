# Safety-ALFRED

## Quickstart

Clone repo:
```bash
$ git clone git@github.com:philipwzf/safety-alfred.git
```

Install requirements(conda):
```bash
$ conda create -n ai2thor python==3.10
$ conda activate ai2thor
$ pip install -r requirements.txt
```

Train models:
```bash
# Setup API_KEY (default is openrouter api)
$ export API_KEY="your_api_key_here"
$ python models/eval/eval_llm_step.py \
    --traj_file data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-305/trial_T20190908_082723_323728/traj_data.json --debug

$ python models/eval/eval_llm_astar.py --debug --traj_file=data/json_2.1.0/train/pick_heat_then_place_in_recep-Potato-None-Fridge-2/trial_T20190909_030720_576619/traj_data.json
```


## Headless Server
```bash
## Setup Xvfb for AI2-THOR
# Start Xvfb on display :99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +extension RANDR +extension RENDER &
export DISPLAY=:99

# Check if thor works
python scrips/check_thor.py
  ###############
  ## (300, 300, 3)
  ## Everything works!!!

```
**Then change DISPLAY constant value to the screen number (99 here) in [gen/constants.py](gen/constants.py)**

Also, checkout this guide: [Setting up THOR on Google Cloud](https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)

## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

## License

MIT License

