# MLP React Flow Solveit

An interactive React Flow laboratory for building, training, and inspecting small multilayer perceptrons in Solveit.

## Run the project

1. Create or open an empty folder in a Solveit instance.
2. Clone this repository into that folder:

   ```bash
   git clone git@github.com:ExploringML/mlp-react-flow-solveit.git .
   ```

3. Open `Main.ipynb` in Solveit.
4. Click **Run all** and wait for setup and the production build to finish.
5. Use the live app embedded near the end of the dialog.

The first run installs the React dependencies and prepares the Vite app. Later **Run all** executions preserve edits under `my-react-flow-app/src/`.

## Reset the app

To rebuild the default app from scratch, delete both:

- `my-react-flow-app/`
- `.solveit-react-flow.json`

Then run all cells in `Main.ipynb` again.
