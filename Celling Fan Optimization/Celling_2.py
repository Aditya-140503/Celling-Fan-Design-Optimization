import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize
import tkinter as tk
from tkinter import messagebox

# Load your dataset
data = pd.read_csv(r"F:\IPR Research Papers\Augmented_Celling_Fan_Data.csv")

# Drop "Design Number" column if not needed
data = data.drop(columns=["Design Number"])


# Prepare features and targets
X = data[['Forward Swept (A)', 'Root Angle of Attack (B)', 'Tips Angle of Attack (C)', 'Tip Width (C)']]
y_flow_rate = data['Volumetric Flow Rate (m^3/min)']
y_torque = data['Torque (N.m)']
y_efficiency = data['Energy Efficiency (m^2/N.min)']

# Fit models
model_flow_rate = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=200)
model_flow_rate.fit(X, y_flow_rate)

model_torque = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=200)
model_torque.fit(X, y_torque)

model_efficiency = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=200)
model_efficiency.fit(X, y_efficiency)

# Additional models
model_decision_tree_flow_rate = DecisionTreeRegressor()
model_random_forest_flow_rate = RandomForestRegressor(n_estimators=100)

model_decision_tree_flow_rate.fit(X, y_flow_rate)
model_random_forest_flow_rate.fit(X, y_flow_rate)

# Train Decision Tree and Random Forest for Torque
model_decision_tree_torque = DecisionTreeRegressor()
model_random_forest_torque = RandomForestRegressor(n_estimators=100)

model_decision_tree_torque.fit(X, y_torque)
model_random_forest_torque.fit(X, y_torque)

# Train Decision Tree and Random Forest for Energy Efficiency
model_decision_tree_efficiency = DecisionTreeRegressor()
model_random_forest_efficiency = RandomForestRegressor(n_estimators=100)

model_decision_tree_efficiency.fit(X, y_efficiency)
model_random_forest_efficiency.fit(X, y_efficiency)



# Define outputs as features for reverse modeling
X_reverse = data[['Volumetric Flow Rate (m^3/min)', 'Torque (N.m)', 'Energy Efficiency (m^2/N.min)']]

# Define each input parameter as a target for reverse modeling
y_tip_width = data['Tip Width (C)']
y_root_angle = data['Root Angle of Attack (B)']
y_tips_angle = data['Tips Angle of Attack (C)']
y_forward_swept = data['Forward Swept (A)']

# Train models for Tip Width
model_tip_width = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=200)
model_decision_tree_tip_width = DecisionTreeRegressor()
model_random_forest_tip_width = RandomForestRegressor(n_estimators=100)

model_tip_width.fit(X_reverse, y_tip_width)
model_decision_tree_tip_width.fit(X_reverse, y_tip_width)
model_random_forest_tip_width.fit(X_reverse, y_tip_width)

# Train models for Root Angle of Attack
model_root_angle = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=200)
model_decision_tree_root_angle = DecisionTreeRegressor()
model_random_forest_root_angle = RandomForestRegressor(n_estimators=100)

model_root_angle.fit(X_reverse, y_root_angle)
model_decision_tree_root_angle.fit(X_reverse, y_root_angle)
model_random_forest_root_angle.fit(X_reverse, y_root_angle)

# Train models for Tips Angle of Attack
model_tip_angle = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=200)
model_decision_tree_tip_angle = DecisionTreeRegressor()
model_random_forest_tip_angle = RandomForestRegressor(n_estimators=100)

model_tip_angle.fit(X_reverse, y_tips_angle)
model_decision_tree_tip_angle.fit(X_reverse, y_tips_angle)
model_random_forest_tip_angle.fit(X_reverse, y_tips_angle)

# Train models for Forward Swept
model_forward_swept = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=200)
model_decision_tree_forward_swept = DecisionTreeRegressor()
model_random_forest_forward_swept = RandomForestRegressor(n_estimators=100)

model_forward_swept.fit(X_reverse, y_forward_swept)
model_decision_tree_forward_swept.fit(X_reverse, y_forward_swept)
model_random_forest_forward_swept.fit(X_reverse, y_forward_swept)


# Function to optimize parameters
def objective_function(x):
    X_new = pd.DataFrame([x], columns=['Forward Swept (A)', 'Root Angle of Attack (B)', 'Tips Angle of Attack (C)', 'Tip Width (C)'])

    # Flow rate predictions
    flow_rate_preds = [model_flow_rate.predict(X_new)[0],
                       model_decision_tree_flow_rate.predict(X_new)[0],
                       model_random_forest_flow_rate.predict(X_new)[0]]

    # Torque predictions
    torque_preds = [model_torque.predict(X_new)[0],
                    model_decision_tree_torque.predict(X_new)[0],
                    model_random_forest_torque.predict(X_new)[0]]

    # Efficiency predictions
    efficiency_preds = [model_efficiency.predict(X_new)[0],
                        model_decision_tree_efficiency.predict(X_new)[0],
                        model_random_forest_efficiency.predict(X_new)[0]]

    # Aggregate predictions
    flow_rate_pred = np.mean(flow_rate_preds)
    torque_pred = np.mean(torque_preds)
    efficiency_pred = np.mean(efficiency_preds)

    # Objective function (e.g., minimizing torque, maximizing flow rate and efficiency)
    objective = - flow_rate_pred + torque_pred - 2 * efficiency_pred
    return objective


def optimize_parameters(user_input):
    initial_sample = pd.Series(user_input).values
    bounds = [(X['Forward Swept (A)'].min(), X['Forward Swept (A)'].max()),
              (X['Root Angle of Attack (B)'].min(), X['Root Angle of Attack (B)'].max()),
              (X['Tips Angle of Attack (C)'].min(), X['Tips Angle of Attack (C)'].max()),
              (X['Tip Width (C)'].min(), X['Tip Width (C)'].max())]

    result = minimize(objective_function, initial_sample, bounds=bounds)
    optimal_parameters = result.x

    X_optimal = pd.DataFrame([optimal_parameters],
                             columns=['Forward Swept (A)', 'Root Angle of Attack (B)', 'Tips Angle of Attack (C)',
                                      'Tip Width (C)'])
    flow_rate_optimal = model_flow_rate.predict(X_optimal)[0]
    torque_optimal = model_torque.predict(X_optimal)[0]
    efficiency_optimal = model_efficiency.predict(X_optimal)[0]

    return {
        "Optimal Parameters": optimal_parameters,
        "Optimized Flow Rate": flow_rate_optimal,
        "Optimized Torque": torque_optimal,
        "Optimized Efficiency": efficiency_optimal
    }


def reverse_objective_function(x, target_output_list):
    X_new = pd.DataFrame([x], columns=['Volumetric Flow Rate (m^3/min)', 'Torque (N.m)', 'Energy Efficiency (m^2/N.min)'])

    # Predict blade geometry parameters
    tip_width_pred = np.mean([
        model_tip_width.predict(X_new)[0],
        model_decision_tree_tip_width.predict(X_new)[0],
        model_random_forest_tip_width.predict(X_new)[0]
    ])

    tip_angle_pred = np.mean([
        model_tip_angle.predict(X_new)[0],
        model_decision_tree_tip_angle.predict(X_new)[0],
        model_random_forest_tip_angle.predict(X_new)[0]
    ])

    root_angle_pred = np.mean([
        model_root_angle.predict(X_new)[0],
        model_decision_tree_root_angle.predict(X_new)[0],
        model_random_forest_root_angle.predict(X_new)[0]
    ])

    forward_swept_pred = np.mean([
        model_forward_swept.predict(X_new)[0],
        model_decision_tree_forward_swept.predict(X_new)[0],
        model_random_forest_forward_swept.predict(X_new)[0]
    ])

    # Extract target values
    target_flow_rate, target_torque, target_efficiency = target_output_list

    # Compute MSE
    mse = np.mean([
        (tip_width_pred - target_flow_rate) ** 2,
        (tip_angle_pred - target_torque) ** 2,
        (root_angle_pred - target_efficiency) ** 2,
        (forward_swept_pred - target_efficiency) ** 2
    ])

    # Apply a weight factor to encourage changes
    weighted_mse = 10 * mse  # Increase weight to force optimization

    # Objective function
    objective = (- tip_width_pred + root_angle_pred - 2 * tip_angle_pred + forward_swept_pred) + weighted_mse

    return objective


def reverse_optimize_parameters(user_output):
    # Convert dictionary values to a list
    target_output_list = [
        user_output['Volumetric Flow Rate'],
        user_output['Torque'],
        user_output['Energy Efficiency']
    ]

    # Convert dictionary to an array for optimization
    initial_sample = np.array(target_output_list)

    # Define bounds
    bounds = [
        (X_reverse['Volumetric Flow Rate (m^3/min)'].min(), X_reverse['Volumetric Flow Rate (m^3/min)'].max()),
        (X_reverse['Torque (N.m)'].min(), X_reverse['Torque (N.m)'].max()),
        (X_reverse['Energy Efficiency (m^2/N.min)'].min(), X_reverse['Energy Efficiency (m^2/N.min)'].max())
    ]

    # **Increase randomness in initial guesses to 15-20%**
    perturbed_sample = initial_sample * (1 + np.random.uniform(-0.01, 0.05, size=len(initial_sample)))

    result = minimize(reverse_objective_function, perturbed_sample, args=(target_output_list,), bounds=bounds)

    if not result.success:
        return {"error": "Optimization failed", "message": result.message}

    optimal_geometry = result.x
    X_optimal = pd.DataFrame([optimal_geometry], columns=[
        'Volumetric Flow Rate (m^3/min)', 'Torque (N.m)', 'Energy Efficiency (m^2/N.min)'
    ])

    # Predict optimal blade geometry
    tip_width_optimal = model_tip_width.predict(X_optimal)[0]
    root_angle_optimal = model_root_angle.predict(X_optimal)[0]
    tips_angle_optimal = model_tip_angle.predict(X_optimal)[0]
    forward_swept_optimal = model_forward_swept.predict(X_optimal)[0]

    # Recompute efficiency results for the optimized design
    X_final = pd.DataFrame([[forward_swept_optimal, root_angle_optimal, tips_angle_optimal, tip_width_optimal]],
                           columns=['Forward Swept (A)', 'Root Angle of Attack (B)', 'Tips Angle of Attack (C)',
                                    'Tip Width (C)'])

    optimized_flow_rate = model_flow_rate.predict(X_final)[0]
    optimized_torque = model_torque.predict(X_final)[0]
    optimized_efficiency = model_efficiency.predict(X_final)[0]

    return {
        "Optimal Blade Geometry": {
            "Tip Width (C)": tip_width_optimal,
            "Root Angle of Attack (B)": root_angle_optimal,
            "Tips Angle of Attack (C)": tips_angle_optimal,
            "Forward Swept (A)": forward_swept_optimal
        },
        "Optimized Flow Rate": optimized_flow_rate,
        "Optimized Torque": optimized_torque,
        "Optimized Efficiency": optimized_efficiency
    }




class FanOptimizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Ceiling Fan Optimization")
        master.geometry("1200x720")
        master.configure(bg='#f0f0f0')

        # Center the window on screen
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 700) // 2
        master.geometry(f"800x700+{x}+{y}")

        # Show the introductory screen
        self.show_intro_screen()


    def add_logos(self):
        try:
            # Load and scale the first logo (top-left corner)
            left_logo_path = r"F:\IPR Research Papers\Crompton.png"
            left_logo_image = tk.PhotoImage(file=left_logo_path)
            scaled_left_logo = left_logo_image.subsample(1, 1)  # Adjust scaling factor as needed
            left_logo_label = tk.Label(self.master, image=scaled_left_logo, bg='#2c3e50')
            left_logo_label.image = scaled_left_logo  # Keep a reference to avoid garbage collection
            left_logo_label.place(x=30, y=10)  # Adjust position as needed for the left logo

            # Load and scale the second logo (top-right corner)
            right_logo_path = r"F:\IPR Research Papers\VIT_logo.png"
            right_logo_image = tk.PhotoImage(file=right_logo_path)
            scaled_right_logo = right_logo_image.subsample(3, 3)  # Adjust scaling factor as needed
            right_logo_label = tk.Label(self.master, image=scaled_right_logo, bg='#2c3e50')
            right_logo_label.image = scaled_right_logo  # Keep a reference to avoid garbage collection
            right_logo_label.place(x=30, y=10)  # Temporary position; will be updated dynamically

            # Function to update the position of the right logo dynamically
            def update_logo_position(event=None):
                window_width = self.master.winfo_width()
                logo_width = scaled_right_logo.width()
                if right_logo_label.winfo_exists():
                    right_logo_label.place(x=window_width - logo_width - 30, y=10)

            # Bind the update function to the Configure event
            self.master.bind("<Configure>", update_logo_position)

            # Initial placement
            update_logo_position()

        except Exception as e:
            print(f"Error loading or scaling logo image: {e}")

    def show_intro_screen(self):
        # Clear previous widgets
        for widget in self.master.winfo_children():
            widget.destroy()

        # Create main frame with padding
        main_frame = tk.Frame(self.master, bg='#f0f0f0', padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Frame with gradient-like effect
        header_frame = tk.Frame(main_frame, bg='#2c3e50', pady=20)
        header_frame.pack(fill=tk.X, pady=(50, 20))

        # Add logos above the title bar
        self.add_logos()

        # Project Title with better formatting
        title_label = tk.Label(header_frame,
                               text="Ceiling Fan Design Optimization",
                               font=("Helvetica", 32, "bold"),
                               fg='white',
                               bg='#2c3e50')
        title_label.pack()

        subtitle_label = tk.Label(header_frame,
                                  text="using Machine Learning Techniques",
                                  font=("Helvetica", 16, "italic"),
                                  fg='#bdc3c7',
                                  bg='#2c3e50')
        subtitle_label.pack()

        # Company Info Frame
        company_frame = tk.Frame(main_frame, bg='#f0f0f0', relief=tk.RIDGE, bd=2)
        company_frame.pack(fill=tk.X, pady=15, padx=20)

        tk.Label(company_frame,
                 text="Sponsor Company",
                 font=("Helvetica", 12, "bold"),
                 fg='#7f8c8d',
                 bg='#f0f0f0').pack(pady=(10, 0))

        tk.Label(company_frame,
                 text="Crompton Greaves Consumer Electricals Limited",
                 font=("Helvetica", 24, "bold"),
                 fg='#2c3e50',
                 bg='#f0f0f0').pack(pady=(0, 10))

        # Mentors Frame
        mentors_frame = tk.Frame(main_frame, bg='#f0f0f0')
        mentors_frame.pack(fill=tk.X, pady=15)

        # Industrial Mentor
        mentor_frame1 = tk.Frame(mentors_frame, bg='#f0f0f0', relief=tk.RIDGE, bd=2)
        mentor_frame1.pack(side=tk.LEFT, expand=True, padx=10, fill=tk.BOTH)

        tk.Label(mentor_frame1,
                 text="Industry Mentor",
                 font=("Helvetica", 12, "bold"),
                 fg='#7f8c8d',
                 bg='#f0f0f0').pack(pady=(10, 0))

        tk.Label(mentor_frame1,
                 text="Pradeep Tonge",
                 font=("Helvetica", 16),
                 fg='#2c3e50',
                 bg='#f0f0f0').pack(pady=(0, 10))

        # Project Guide
        mentor_frame2 = tk.Frame(mentors_frame, bg='#f0f0f0', relief=tk.RIDGE, bd=2)
        mentor_frame2.pack(side=tk.LEFT, expand=True, padx=10, fill=tk.BOTH)

        tk.Label(mentor_frame2,
                 text="Project Guide",
                 font=("Helvetica", 12, "bold"),
                 fg='#7f8c8d',
                 bg='#f0f0f0').pack(pady=(10, 0))

        tk.Label(mentor_frame2,
                 text="Ratnmala Bhimanpallewar",
                 font=("Helvetica", 16),
                 fg='#2c3e50',
                 bg='#f0f0f0').pack(pady=(0, 10))

        # Team Frame
        team_frame = tk.Frame(main_frame, bg='#f0f0f0', relief=tk.RIDGE, bd=2)
        team_frame.pack(fill=tk.X, pady=15, padx=20)

        tk.Label(team_frame,
                 text="Project Team",
                 font=("Helvetica", 12, "bold"),
                 fg='#7f8c8d',
                 bg='#f0f0f0').pack(pady=(10, 5))

        # Team members in a grid
        members = ["Aditya", "Sarvesh", "Gunesh", "Arya"]
        member_frame = tk.Frame(team_frame, bg='#f0f0f0')
        member_frame.pack(pady=(0, 10))

        for i, member in enumerate(members):
            tk.Label(member_frame,
                     text=member,
                     font=("Helvetica", 16),
                     fg='#2c3e50',
                     bg='#f0f0f0',
                     width=10).pack(side=tk.LEFT, padx=5)

        # Add the fan image with larger size and minimal padding
        try:
            original_image = tk.PhotoImage(file=r"C:\Users\Acer\Downloads\Fan2.png")

            # Larger maximum dimensions
            max_width = 300
            max_height = 300

            # Calculate scaling factor
            width_ratio = max_width / original_image.width()
            height_ratio = max_height / original_image.height()
            scale_factor = min(width_ratio, height_ratio)

            # Calculate new dimensions
            new_width = int(original_image.width() * scale_factor)
            new_height = int(original_image.height() * scale_factor)

            subsample_ratio = int(1 / scale_factor)
            fan_image = original_image.subsample(subsample_ratio, subsample_ratio)

            fan_image_label = tk.Label(main_frame, image=fan_image, bg='#f0f0f0')
            fan_image_label.image = fan_image
            fan_image_label.pack(pady=(10, 10))  # Added small padding to separate from button
        except Exception as e:
            print(f"Error loading/resizing image: {e}")

        # Next Button with hover effect
        self.next_button = tk.Button(main_frame,
                                     text="Start Optimization →",
                                     font=("Helvetica", 14, "bold"),
                                     command=self.show_optimization_screen,
                                     bg='#2ecc71',
                                     fg='white',
                                     relief=tk.FLAT,
                                     padx=20,
                                     pady=10,
                                     cursor="hand2")
        self.next_button.pack(pady=(0, 30))  # Adjusted padding for bottom margin

        # Bind hover events
        self.next_button.bind("<Enter>", lambda e: e.widget.config(bg='#27ae60'))
        self.next_button.bind("<Leave>", lambda e: e.widget.config(bg='#2ecc71'))

    def update_input_fields(self):
        """Update the input fields dynamically based on the selected mode without refreshing the entire screen."""
        # Clear existing input fields
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        # Define styles
        input_style = {'font': ("Helvetica", 11), 'bg': 'white', 'fg': '#2c3e50', 'relief': tk.SOLID, 'bd': 1}
        label_style = {'font': ("Helvetica", 11), 'bg': '#f0f0f0', 'fg': '#2c3e50'}

        # Define parameters based on mode
        params = {
            "design_to_efficiency": [
                ("Tip Width (C):", "tip_width_entry"),
                ("Tips Angle of Attack (C):", "tip_angle_entry"),
                ("Root Angle of Attack (B):", "root_angle_entry"),
                ("Forward Swept (A):", "forward_swept_entry")
            ],
            "efficiency_to_design": [
                ("Volumetric Flow Rate:", "flow_rate_entry"),
                ("Torque:", "torque_entry"),
                ("Energy Efficiency:", "efficiency_entry")
            ]
        }

        selected_params = params[self.mode.get()]

        # Recreate input fields
        for label_text, attr_name in selected_params:
            frame = tk.Frame(self.input_frame, bg='#f0f0f0')
            frame.pack(fill=tk.X, padx=20, pady=5)

            tk.Label(frame, text=label_text, **label_style).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=20, **input_style)
            entry.pack(side=tk.RIGHT)
            setattr(self, f"{attr_name}", entry)

    def show_optimization_screen(self):
        # Clear previous widgets
        for widget in self.master.winfo_children():
            widget.destroy()

        # Create main frame with gradient-like background
        main_frame = tk.Frame(self.master, bg='#f0f0f0', padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg='#2c3e50', pady=15)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(header_frame,
                 text="Fan Parameter Optimization",
                 font=("Helvetica", 24, "bold"),
                 fg='white',
                 bg='#2c3e50').pack()

        # Mode Selection Frame
        mode_frame = tk.Frame(main_frame, bg='#f0f0f0', pady=10)
        mode_frame.pack(fill=tk.X)

        tk.Label(mode_frame, text="Select Mode:", font=("Helvetica", 12, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(
            side=tk.LEFT, padx=10)

        self.mode = tk.StringVar(value="design_to_efficiency")  # Default mode

        design_mode_button = tk.Radiobutton(mode_frame, text="Design to Efficiency",
                                            variable=self.mode, value="design_to_efficiency",
                                            font=("Helvetica", 11), bg='#f0f0f0', fg='#2c3e50',
                                            command=self.update_input_fields)
        design_mode_button.pack(side=tk.LEFT, padx=5)

        efficiency_mode_button = tk.Radiobutton(mode_frame, text="Efficiency to Design",
                                                variable=self.mode, value="efficiency_to_design",
                                                font=("Helvetica", 11), bg='#f0f0f0', fg='#2c3e50',
                                                command=self.update_input_fields)
        efficiency_mode_button.pack(side=tk.LEFT, padx=5)

        # Create two columns
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Input Section (Left Frame)
        self.input_frame = tk.LabelFrame(left_frame, text="Input Parameters",
                                         font=("Helvetica", 12, "bold"), bg='#f0f0f0', fg='#2c3e50', pady=10)
        self.input_frame.pack(fill=tk.X, pady=10)

        # Update input fields dynamically
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        # Style for input fields
        input_style = {'font': ("Helvetica", 11), 'bg': 'white', 'fg': '#2c3e50', 'relief': tk.SOLID, 'bd': 1}
        label_style = {'font': ("Helvetica", 11), 'bg': '#f0f0f0', 'fg': '#2c3e50'}

        # Parameters
        params = {
            "design_to_efficiency": [
                ("Tip Width (C):", "tip_width_entry"),
                ("Tips Angle of Attack (C):", "tip_angle_entry"),
                ("Root Angle of Attack (B):", "root_angle_entry"),
                ("Forward Swept (A):", "forward_swept_entry")
            ],
            "efficiency_to_design": [
                ("Volumetric Flow Rate:", "flow_rate_entry"),
                ("Torque:", "torque_entry"),
                ("Energy Efficiency:", "efficiency_entry")
            ]
        }

        selected_params = params[self.mode.get()]

        for label_text, attr_name in selected_params:
            frame = tk.Frame(self.input_frame, bg='#f0f0f0')
            frame.pack(fill=tk.X, padx=20, pady=5)

            tk.Label(frame, text=label_text, **label_style).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=20, **input_style)
            entry.pack(side=tk.RIGHT)
            setattr(self, f"{attr_name}", entry)

        # Buttons
        button_frame = tk.Frame(left_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=20)

        self.optimize_button = tk.Button(button_frame,
                                         text="Optimize Parameters →",
                                         font=("Helvetica", 12, "bold"),
                                         bg='#2ecc71',
                                         fg='white',
                                         relief=tk.FLAT,
                                         padx=20,
                                         pady=8,
                                         cursor="hand2",
                                         command=self.optimize)
        self.optimize_button.pack(side=tk.LEFT, padx=5)

        self.use_optimized_button = tk.Button(button_frame,
                                              text="Use Optimized Parameters",
                                              font=("Helvetica", 12, "bold"),
                                              bg='#3498db',
                                              fg='white',
                                              relief=tk.FLAT,
                                              padx=20,
                                              pady=8,
                                              cursor="hand2",
                                              command=self.use_optimized_parameters)
        self.use_optimized_button.pack(side=tk.LEFT, padx=5)

        back_button = tk.Button(button_frame,
                                text="← Back",
                                font=("Helvetica", 12),
                                bg='#e74c3c',
                                fg='white',
                                relief=tk.FLAT,
                                padx=20,
                                pady=8,
                                cursor="hand2",
                                command=self.show_intro_screen)
        back_button.pack(side=tk.LEFT, padx=5)

        # Add hover effects for buttons
        for button in [self.optimize_button, back_button, self.use_optimized_button]:
            button.bind("<Enter>",
                        lambda e, b=button: b.config(
                            bg='#27ae60' if b == self.optimize_button else '#c0392b' if b == back_button else '#2980b9'))
            button.bind("<Leave>",
                        lambda e, b=button: b.config(
                            bg='#2ecc71' if b == self.optimize_button else '#e74c3c' if b == back_button else '#3498db'))

        # Add the fan image below the buttons
        try:
            original_image = tk.PhotoImage(file=r"C:\Users\Acer\Downloads\Fan2.png")

            # Resize image
            max_width = 300
            max_height = 300
            width_ratio = max_width / original_image.width()
            height_ratio = max_height / original_image.height()
            scale_factor = min(width_ratio, height_ratio)

            subsample_ratio = int(1 / scale_factor)
            fan_image = original_image.subsample(subsample_ratio, subsample_ratio)

            fan_image_label = tk.Label(left_frame, image=fan_image,
                                       bg='#f0f0f0')  # Use left_frame for correct placement
            fan_image_label.image = fan_image
            fan_image_label.pack(pady=(10, 10))  # Add padding below buttons
        except Exception as e:
            print(f"Error loading/resizing image: {e}")

        # Results Section (Right Frame)
        results_frame = tk.LabelFrame(right_frame,
                                      text="Optimization Results",
                                      font=("Helvetica", 12, "bold"),
                                      bg='#f0f0f0',
                                      fg='#2c3e50')
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(results_frame,
                                    height=25,
                                    font=("Helvetica", 11),
                                    wrap="word",
                                    padx=10,
                                    pady=10,
                                    bg='white',
                                    fg='#2c3e50',
                                    relief=tk.SOLID)
        self.results_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)

    def use_optimized_parameters(self):
        """Updates input fields based on selected mode and optimized parameters"""
        if hasattr(self, 'optimized_parameters') and self.optimized_parameters:
            print("DEBUG - Using optimized parameters:", self.optimized_parameters)
            print("DEBUG - Current Mode:", self.mode.get())

            if self.mode.get() == 'design_to_efficiency':
                print("DEBUG - Updating Design Fields")  # Fixed this!

                if "Optimal Parameters" not in self.optimized_parameters:
                    messagebox.showwarning("Missing Data", "Optimized parameters not found!")
                    return

                optimized_values = self.optimized_parameters["Optimal Parameters"]

                # Ensure it's a list, not NumPy array
                if isinstance(optimized_values, np.ndarray):
                    optimized_values = optimized_values.tolist()

                print("DEBUG - Optimized Values:", optimized_values)

                # Ensure each value is a Python float and update the fields
                self.tip_width_entry.delete(0, tk.END)
                self.tip_width_entry.insert(0, f"{float(optimized_values[3]):.3f}")

                self.tip_angle_entry.delete(0, tk.END)
                self.tip_angle_entry.insert(0, f"{float(optimized_values[2]):.3f}")

                self.root_angle_entry.delete(0, tk.END)
                self.root_angle_entry.insert(0, f"{float(optimized_values[1]):.3f}")

                self.forward_swept_entry.delete(0, tk.END)
                self.forward_swept_entry.insert(0, f"{float(optimized_values[0]):.3f}")

            else:  # Efficiency to Design Mode
                print("DEBUG - Updating Efficiency Fields")

                if "Optimized Flow Rate" not in self.optimized_parameters:
                    messagebox.showwarning("Missing Data", "Optimized flow rate not found!")
                    return

                # Convert values to standard float before inserting
                self.flow_rate_entry.delete(0, tk.END)
                self.flow_rate_entry.insert(0, f"{float(self.optimized_parameters['Optimized Flow Rate']):.3f}")

                self.torque_entry.delete(0, tk.END)
                self.torque_entry.insert(0, f"{float(self.optimized_parameters['Optimized Torque']):.3f}")

                self.efficiency_entry.delete(0, tk.END)
                self.efficiency_entry.insert(0, f"{float(self.optimized_parameters['Optimized Efficiency']):.3f}")

            # Force UI refresh
            self.master.update_idletasks()

            messagebox.showinfo("Parameters Updated", "Input fields updated with optimized parameters.")
            self.optimize()

        else:
            messagebox.showwarning("No Optimized Parameters", "Please run optimization first.")

    def optimize(self):
        try:
            print("Optimize function started!")  # Debugging
            print("Current Mode:", self.mode.get())

            # DESIGN TO EFFICIENCY MODE
            if self.mode.get() == "design_to_efficiency":
                # Ensure all design inputs are filled
                if (self.tip_width_entry.get() == "" or
                        self.tip_angle_entry.get() == "" or
                        self.root_angle_entry.get() == "" or
                        self.forward_swept_entry.get() == ""):
                    raise ValueError("All design parameters must be filled.")

                user_input = {
                    'Tip Width (C)': float(self.tip_width_entry.get()),
                    'Tips Angle of Attack (C)': float(self.tip_angle_entry.get()),
                    'Root Angle of Attack (B)': float(self.root_angle_entry.get()),
                    'Forward Swept (A)': float(self.forward_swept_entry.get())
                }

                print("User Input:", user_input)  # Debugging

                # Ensure positive values
                if any(value <= 0 for value in user_input.values()):
                    raise ValueError("All design inputs must be positive values.")

                X_initial = pd.DataFrame([user_input], columns=[
                    'Forward Swept (A)', 'Root Angle of Attack (B)', 'Tips Angle of Attack (C)', 'Tip Width (C)'
                ])

                # Get model predictions
                flow_rate_initial = model_flow_rate.predict(X_initial)[0]
                torque_initial = model_torque.predict(X_initial)[0]
                efficiency_initial = model_efficiency.predict(X_initial)[0]

                result = optimize_parameters(user_input)

                # Store optimized parameters but DO NOT update UI yet
                self.optimized_parameters = {
                    "Optimal Parameters": result['Optimal Parameters'].tolist(),  # Convert NumPy array to list
                    "Optimized Flow Rate": float(result["Optimized Flow Rate"]),
                    "Optimized Torque": float(result["Optimized Torque"]),
                    "Optimized Efficiency": float(result["Optimized Efficiency"])
                }

                print("DEBUG - Stored Optimized Parameters:", self.optimized_parameters)  # Debugging

                results_text = (
                    f"Initial Inputs:\n"
                    f"Tip Width (C): {user_input['Tip Width (C)']}\n"
                    f"Tips Angle of Attack (C): {user_input['Tips Angle of Attack (C)']}\n"
                    f"Root Angle of Attack (B): {user_input['Root Angle of Attack (B)']}\n"
                    f"Forward Swept (A): {user_input['Forward Swept (A)']}\n\n"
                    f"Predicted Results for Initial Inputs:\n"
                    f"Volumetric Flow Rate: {flow_rate_initial:.2f} m^3/min\n"
                    f"Torque: {torque_initial:.2f} N.m\n"
                    f"Energy Efficiency: {efficiency_initial:.2f} m^2/N.min\n\n"
                    f"Optimization Results:\n"
                    f"Tip Width (C): {result['Optimal Parameters'][3]:.2f}\n"
                    f"Tips Angle of Attack (C): {result['Optimal Parameters'][2]:.2f}\n"
                    f"Root Angle of Attack (B): {result['Optimal Parameters'][1]:.2f}\n"
                    f"Forward Swept (A): {result['Optimal Parameters'][0]:.2f}\n\n"
                    f"Optimized Results:\n"
                    f"Volumetric Flow Rate: {result['Optimized Flow Rate']:.2f} m^3/min\n"
                    f"Torque: {result['Optimized Torque']:.2f} N.m\n"
                    f"Energy Efficiency: {result['Optimized Efficiency']:.2f} m^2/N.min\n"
                )

            # EFFICIENCY TO DESIGN MODE
            elif self.mode.get() == "efficiency_to_design":
                # Ensure all efficiency inputs are filled
                if (self.flow_rate_entry.get() == "" or
                        self.torque_entry.get() == "" or
                        self.efficiency_entry.get() == ""):
                    raise ValueError("All efficiency parameters must be filled.")

                user_output = {
                    'Volumetric Flow Rate': float(self.flow_rate_entry.get()),
                    'Torque': float(self.torque_entry.get()),
                    'Energy Efficiency': float(self.efficiency_entry.get())
                }

                print("User Output:", user_output)  # Debugging

                X_initial = pd.DataFrame([user_output], columns=['Volumetric Flow Rate', 'Torque', 'Energy Efficiency'])

                # Rename columns to match the model's expected feature names
                X_initial.rename(columns={
                    'Volumetric Flow Rate': 'Volumetric Flow Rate (m^3/min)',
                    'Torque': 'Torque (N.m)',
                    'Energy Efficiency': 'Energy Efficiency (m^2/N.min)'
                }, inplace=True)

                # Get model predictions
                tip_width_initial = model_tip_width.predict(X_initial)[0]
                tip_angle_initial = model_tip_angle.predict(X_initial)[0]
                root_angle_initial = model_root_angle.predict(X_initial)[0]
                forward_swept_initial = model_forward_swept.predict(X_initial)[0]

                result = reverse_optimize_parameters(user_output)

                # Store optimized parameters but DO NOT update UI yet
                self.optimized_parameters = result

                print("DEBUG - Stored Optimized Parameters:", self.optimized_parameters)  # Debugging

                results_text = (
                    f"Efficiency to Design Mode:\n"
                    f"Initial Inputs:\n"
                    f"Volumetric Flow Rate: {user_output['Volumetric Flow Rate']:.2f} m^3/min\n"
                    f"Torque: {user_output['Torque']:.2f} N.m\n"
                    f"Energy Efficiency: {user_output['Energy Efficiency']:.2f} m^2/N.min\n\n"

                    f"Predicted Results for Initial Inputs:\n"
                    f"Tip Width (C): {tip_width_initial:.2f}\n"
                    f"Tips Angle of Attack (C): {tip_angle_initial:.2f}\n"
                    f"Root Angle of Attack (B): {root_angle_initial:.2f}\n"
                    f"Forward Swept (A): {forward_swept_initial:.2f}\n\n"
                    
                    f"Optimization Results:\n"
                    f"Tip Width (C): {result['Optimal Blade Geometry']['Tip Width (C)']:.2f}\n"
                    f"Tips Angle of Attack (C): {result['Optimal Blade Geometry']['Tips Angle of Attack (C)']:.2f}\n"
                    f"Root Angle of Attack (B): {result['Optimal Blade Geometry']['Root Angle of Attack (B)']:.2f}\n"
                    f"Forward Swept (A): {result['Optimal Blade Geometry']['Forward Swept (A)']:.2f}\n\n"


                    f"Optimized Results:\n"
                    f"Volumetric Flow Rate: {result['Optimized Flow Rate']:.2f} m^3/min\n"
                    f"Torque: {result['Optimized Torque']:.2f} N.m\n"
                    f"Energy Efficiency: {result['Optimized Efficiency']:.2f} m^2/N.min\n"
                )

            # **DO NOT UPDATE INPUT FIELDS HERE**
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, results_text)
            self.results_text.config(state=tk.DISABLED)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))


# Initialize the app
root = tk.Tk()
app = FanOptimizationApp(root)
root.mainloop()