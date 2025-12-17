# Machine Learning Training Visualizer

Interactive Streamlit project that demonstrates how different machine learning models learn over time. Use sliders and controls to explore gradient descent, loss minimisation, and evolving decision boundaries across linear regression, logistic regression, and a compact neural network.

## Features
- Forward propagation snapshots for three model families (linear, logistic, neural network)
- Adjustable learning rate, epoch budget, and training controls (play/pause, next epoch, reset)
- Scatter plots of training/testing data, regression line fitting, and classification decision boundaries
- Live loss and accuracy trends and numeric metrics per epoch
- Optional neural network diagram showing weight magnitudes and directionality during training

## Getting Started
1. Create and activate a virtual environment (recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```powershell
   streamlit run app.py
   ```
4. Enter the local URL printed in the terminal to interact with the visualisations.

## Usage Tips
- Select a model in the sidebar to switch between regression and classification scenarios.
- Use the learning rate and epoch controls to experiment with convergence speed and stability.
- Click **Play Training** to animate the learning process or **Next Epoch** for frame-by-frame updates.
- Regenerate datasets to observe how different samples influence the optimisation path.
- Watch the loss, accuracy, and decision boundary evolve to understand gradient descent behaviour.

## Project Structure
```
app.py               # Streamlit entry point and interactive layout
training_sim.py      # Lightweight model implementations and dataset factories
visualizations.py    # Plotly figure helpers for data, metrics, and network diagrams
requirements.txt     # Python dependencies
README.md            # Project documentation
```
