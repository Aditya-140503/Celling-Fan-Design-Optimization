import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize
import tkinter as tk
from tkinter import messagebox

# Load your dataset
data = pd.read_csv(r"F:\IPR Research Papers\Celling Fan Data.csv")

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
model_decision_tree = DecisionTreeRegressor()
model_random_forest = RandomForestRegressor(n_estimators=100)

model_decision_tree.fit(X, y_flow_rate)
model_random_forest.fit(X, y_flow_rate)


# Function to optimize parameters
def objective_function(x):
    X_new = pd.DataFrame([x], columns=['Forward Swept (A)', 'Root Angle of Attack (B)', 'Tips Angle of Attack (C)',
                                       'Tip Width (C)'])
    flow_rate_preds = [model_flow_rate.predict(X_new)[0], model_decision_tree.predict(X_new)[0],
                       model_random_forest.predict(X_new)[0]]
    torque_preds = [model_torque.predict(X_new)[0], model_decision_tree.predict(X_new)[0],
                    model_random_forest.predict(X_new)[0]]
    efficiency_preds = [model_efficiency.predict(X_new)[0], model_decision_tree.predict(X_new)[0],
                        model_random_forest.predict(X_new)[0]]

    flow_rate_pred = np.mean(flow_rate_preds)
    torque_pred = np.mean(torque_preds)
    efficiency_pred = np.mean(efficiency_preds)

    objective = - 2 * flow_rate_pred + torque_pred - efficiency_pred
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


class FanOptimizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Ceiling Fan Optimization")
        master.geometry("800x700")
        master.configure(bg='#f0f0f0')

        # Optional: Center the window on screen
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 700) // 2
        master.geometry(f"800x700+{x}+{y}")

        # Show the introductory screen
        self.show_intro_screen()

    def show_intro_screen(self):
        # Clear previous widgets
        for widget in self.master.winfo_children():
            widget.destroy()

        # Create main frame with padding
        main_frame = tk.Frame(self.master, bg='#f0f0f0', padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header Frame with gradient-like effect
        header_frame = tk.Frame(main_frame, bg='#2c3e50', pady=20)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        # Project Title with better formatting
        title_label = tk.Label(header_frame,
                               text="Ceiling Fan Optimization",
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
                 text="Crompton Greaves",
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
                 text="Industrial Mentor",
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
        self.next_button.pack(pady=20)

        # Bind hover events
        self.next_button.bind("<Enter>", lambda e: e.widget.config(bg='#27ae60'))
        self.next_button.bind("<Leave>", lambda e: e.widget.config(bg='#2ecc71'))

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

        # Create two columns
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Input Section (Left Frame)
        input_frame = tk.LabelFrame(left_frame, text="Input Parameters",
                                    font=("Helvetica", 12, "bold"), bg='#f0f0f0', fg='#2c3e50', pady=10)
        input_frame.pack(fill=tk.X, pady=10)

        # Style for input fields
        input_style = {
            'font': ("Helvetica", 11),
            'bg': 'white',
            'fg': '#2c3e50',
            'relief': tk.SOLID,
            'bd': 1
        }

        label_style = {
            'font': ("Helvetica", 11),
            'bg': '#f0f0f0',
            'fg': '#2c3e50'
        }

        # Input fields with better spacing and styling
        params = [
            ("Tip Width (C):", "tip_width"),
            ("Tips Angle of Attack (C):", "tip_angle"),
            ("Root Angle of Attack (B):", "root_angle"),
            ("Forward Swept (A):", "forward_swept")
        ]

        for label_text, attr_name in params:
            frame = tk.Frame(input_frame, bg='#f0f0f0')
            frame.pack(fill=tk.X, padx=20, pady=5)

            tk.Label(frame, text=label_text, **label_style).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=20, **input_style)
            entry.pack(side=tk.RIGHT)
            setattr(self, f"{attr_name}_entry", entry)

        # Buttons with hover effects
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

        # Results Section (Right Frame)
        results_frame = tk.LabelFrame(right_frame,
                                      text="Optimization Results",
                                      font=("Helvetica", 12, "bold"),
                                      bg='#f0f0f0',
                                      fg='#2c3e50')
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results Text Area with better styling
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

        # Add hover effects for buttons
        for button in [self.optimize_button, back_button]:
            button.bind("<Enter>",
                        lambda e, b=button: b.config(bg='#27ae60' if b == self.optimize_button else '#c0392b'))
            button.bind("<Leave>",
                        lambda e, b=button: b.config(bg='#2ecc71' if b == self.optimize_button else '#e74c3c'))

    def optimize(self):
        try:
            # Check for empty inputs
            if (self.tip_width_entry.get() == "" or
                    self.tip_angle_entry.get() == "" or
                    self.root_angle_entry.get() == "" or
                    self.forward_swept_entry.get() == ""):
                raise ValueError("All input fields must be filled.")

            user_input = {
                'Tip Width (C)': float(self.tip_width_entry.get()),
                'Tips Angle of Attack (C)': float(self.tip_angle_entry.get()),
                'Root Angle of Attack (B)': float(self.root_angle_entry.get()),
                'Forward Swept (A)': float(self.forward_swept_entry.get())
            }

            # Additional validation for positive values
            if (user_input['Tip Width (C)'] <= 0 or
                    user_input['Tips Angle of Attack (C)'] < 0 or
                    user_input['Root Angle of Attack (B)'] < 0 or
                    user_input['Forward Swept (A)'] <= 0):
                raise ValueError("All inputs must be positive values.")

            X_initial = pd.DataFrame([user_input], columns=['Forward Swept (A)', 'Root Angle of Attack (B)',
                                                            'Tips Angle of Attack (C)', 'Tip Width (C)'])

            flow_rate_initial = model_flow_rate.predict(X_initial)[0]
            torque_initial = model_torque.predict(X_initial)[0]
            efficiency_initial = model_efficiency.predict(X_initial)[0]

            result = optimize_parameters(user_input)

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
                f"Optimal Parameters:\n"
                f"Tip Width (C): {result['Optimal Parameters'][3]:.2f}\n"
                f"Tips Angle of Attack (C): {result['Optimal Parameters'][2]:.2f}\n"
                f"Root Angle of Attack (B): {result['Optimal Parameters'][1]:.2f}\n"
                f"Forward Swept (A): {result['Optimal Parameters'][0]:.2f}\n\n"
                f"Optimized Results:\n"
                f"Volumetric Flow Rate: {result['Optimized Flow Rate']:.2f} m^3/min\n"
                f"Torque: {result['Optimized Torque']:.2f} N.m\n"
                f"Energy Efficiency: {result['Optimized Efficiency']:.2f} m^2/N.min\n"
            )

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