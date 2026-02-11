The optimization model is handled through Benders decomposition. The planning and scheduling of parking lots are in the master level (multi master), and the power flow is performed in the subproblem.
The main loop of the Benders is in the main file.
The other files come to the main file through functions.
The Excel files are the input to the model, including the arrival and departure times of EVs and the input to the power flow.
