# Data collection
To use the data collection pipeline:
```bash
cd canopies_ws
./bash_scripts/data_collection_routine.sh 
```
Once the simulation and ROS are running, you may press inv when to start the environment.
The following are the commands:
 - trigger (kept press) records the trajectory steps
 - gripper (kept press) disconnects the vr remote from the simulation
 - "A" for saving the recorded trajectory
 - "B" for descarding the recorded steps and restart to record


# Testing the imitation learning agents
To use the testing pipeline:
```bash
cd canopies_ws
./bash_scripts/testing_routine.sh agent_name
```
With:
- `agent_name={agent, agent_stable}` depending which one to test
When launched, the setup routine asks if you want to record the trajectory (press "y") or not (press "n")

