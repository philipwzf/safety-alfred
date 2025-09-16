## Setup API_KEY
export API_KEY="your_api_key_here"

## Setup Xvfb for AI2-THOR
# Start Xvfb on display :99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +extension RANDR +extension RENDER &
export DISPLAY=:99
# Then change DISPLAY constant value to the screen number (99 here) in gen/constants.py
