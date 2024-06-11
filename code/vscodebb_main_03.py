import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import threading
import queue
import cv2
import os
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import multiprocessing

# Global variables to store start time, end time, and duration
start_time = None
end_time = None
duration = None

frame_queue = queue.Queue()
stop_event = threading.Event()

def load_config():
    global source, model_path, threshold, text_size, model, class_names_02
    config_file = "config.txt"
    with open(config_file, "r") as file:
        config = file.readlines()
        source = config[0].strip()
        model_path = config[1].strip()
        threshold = float(config[2].strip())
        text_size = int(config[3].strip())
    model = YOLO(model_path)
    class_names_02 = [f"{key}: {value}" for key, value in model.names.items()]

def load_input_values():
    global operators_value, baskets_value, demand_value, current_temp_value
    input_file = "input.txt"
    with open(input_file, "r") as file:
        input_values = file.readlines()
        operators_value = input_values[0].strip()
        baskets_value = input_values[1].strip()
        demand_value = input_values[2].strip()
        current_temp_value = input_values[3].strip()

load_config()
load_input_values()

def video_processing_thread(source, model, threshold, text_size, selected_product, shared_count):
    class_id, class_name = selected_product.split(": ")
    class_id = int(class_id)
    class_names = {class_id: class_name}
    
    try:
        source = int(source)
        cap = cv2.VideoCapture(source)
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Failed to open video source")
        return

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    start_y = int(h * 0)
    end_y = int(h * 1)
    x_pos = int(w * 0.85)
    region_points = [(x_pos, start_y), (x_pos, end_y)]

    # Create the output directory with the current date if it doesn't exist
    todays_date = datetime.now().strftime("%d%b%Y").upper()
    output_dir = f"output/Date_{todays_date}"
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{current_time}.avi"
    video_writer = cv2.VideoWriter(f"{output_dir}/{output_filename}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    counter = object_counter.ObjectCounter(view_img=True, reg_pts=region_points, classes_names=class_names, draw_tracks=False, line_thickness=2, view_in_counts=False)

    while not stop_event.is_set():
        success, im0 = cap.read()
        if not success:
            break

        # Resize the frame to reduce computation time
        im0 = cv2.resize(im0, (640, 480))
       
        # Perform model inference
        tracks = model.track(im0, persist=True, show=True, conf=threshold)
        
        try:
            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)
            #frame_queue.put(im0)
        except KeyError as e:
            print(f"KeyError: {e}")
            continue
        
        for class_name, counts in counter.class_wise_count.items():
            shared_count.set(str(counts['OUT']))

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

class KitchenDemandApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Kitchen Demand System")
        self.geometry("880x450")
        self.configure(bg="white")

        self.style = ttk.Style(self)
        self.style.configure("TFrame", background="white")
        self.style.configure("Green.TButton", background="green", foreground="green", borderwidth=10, relief="solid")
        self.style.configure("Red.TButton", background="red", foreground="red", borderwidth=10, relief="solid")
        self.style.configure("White.TLabelframe", foreground="black", borderwidth=10)
        self.style.configure("White.TLabelframe.Label", background="white")
        self.style.configure('Custom.TEntry', background='white', foreground='black', relief="solid")

        self.shared_count = tk.StringVar()
        self.shared_count.set("None")

        self.start_time = None  # Initialize start time
        self.end_time = None  # Initialize end time

        self.create_widgets()
        self.update_time()

    def create_widgets(self):
        header_frame = ttk.Frame(self, padding="10")
        header_frame.grid(row=0, column=0, columnspan=8, sticky="ew")

        header_label = ttk.Label(header_frame, text="   Kitchen Demand System    Line -3                 ", font=("Helvetica", 14, "bold"), foreground="yellow", background="black")
        header_label.grid(row=0, column=0, sticky="ew")
        header_label.configure(anchor="w")  # Align text to the left

        self.time_label = ttk.Label(header_frame, text="", font=("Helvetica", 14, "bold"), foreground="yellow", background="black")
        self.time_label.grid(row=0, column=1, sticky="e")

        main_frame = ttk.Frame(self, padding="10", style="Custom.TFrame")
        main_frame.grid(row=1, column=0, columnspan=8, sticky="nsew")

        self.create_form(main_frame, self.shared_count)  # Create the main form

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    def create_form(self, frame, shared_count):
        bounding_box = ttk.LabelFrame(frame, padding="5", labelanchor="n", style="White.TLabelframe")
        bounding_box.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=20, pady=10)

        ttk.Label(bounding_box, text="Operator(s)", font=("Helvetica", 12)).grid(row=1, column=0, sticky="w")
        self.operators = ttk.Entry(bounding_box, width=10, font=("Helvetica", 12), style='Custom.TEntry')
        self.operators.insert(0, operators_value)
        self.operators.grid(row=1, column=1, sticky="w")

        ttk.Label(bounding_box, text="Basket(s)", font=("Helvetica", 12)).grid(row=3, column=0, sticky="w")
        self.baskets = ttk.Entry(bounding_box, width=10, font=("Helvetica", 12))
        self.baskets.insert(0, baskets_value)
        self.baskets.grid(row=3, column=1, sticky="w")

        bounding_box.grid_columnconfigure(2, minsize=50)

        ttk.Label(bounding_box, text="Item Code        ", font=("Helvetica", 12)).grid(row=0, column=3, sticky="w")
        self.item_code = ttk.Entry(bounding_box, font=("Helvetica", 12))
        self.item_code.insert(0, "")
        self.item_code.grid(row=0, column=4, sticky="ew")

        ttk.Label(bounding_box, text="Select Product           ", font=("Helvetica", 12)).grid(row=2, column=3, sticky="w")
        self.select_product = ttk.Combobox(bounding_box, values=class_names_02, font=("Helvetica", 12))
        self.select_product.set(class_names_02[0] if class_names_02 else "No Products")
        self.select_product.grid(row=2, column=4, sticky="ew")

        ttk.Label(bounding_box, text="Demand", font=("Helvetica", 12)).grid(row=4, column=3, sticky="w")
        self.demand = ttk.Entry(bounding_box, font=("Helvetica", 12))
        self.demand.insert(0, demand_value)
        self.demand.grid(row=4, column=4, sticky="ew")

        ttk.Label(bounding_box, text="", font=("Helvetica", 12)).grid(row=5, column=3, sticky="w")

        ttk.Label(bounding_box, text="Current Temp", font=("Helvetica", 12)).grid(row=6, column=3, sticky="w")
        self.current_temp = ttk.Entry(bounding_box, font=("Helvetica", 12))
        self.current_temp.insert(0, current_temp_value)
        self.current_temp.grid(row=6, column=4, sticky="ew")

        ttk.Label(bounding_box, text="", font=("Helvetica", 12)).grid(row=7, column=3, sticky="w")

        bounding_box.grid_columnconfigure(5, minsize=45)
        bounding_box.grid_columnconfigure(6, minsize=45)

        ttk.Label(bounding_box, text="Count", font=("Helvetica", 12), anchor="center").grid(row=1, column=6, sticky="w")
        self.count = ttk.Label(bounding_box, textvariable=shared_count, background="black", foreground="white", font=("Helvetica", 30), width=6, anchor="center")
        self.count.grid(row=2, column=6, sticky="nsew", rowspan=5)
        
        # Dummy rows
        ttk.Label(bounding_box, text="", font=("Helvetica", 12)).grid(row=10, column=3, sticky="w")

        self.start_time_button = ttk.Button(bounding_box, text=" Start  Time  ", command=self.start_detection, style="Green.TButton")
        self.start_time_button.grid(row=11, column=0, columnspan=1, pady=10, sticky="ew")

        self.start_time_label = ttk.Label(bounding_box, text="", font=("Helvetica", 12, "bold"), foreground="green")
        self.start_time_label.grid(row=11, column=1, sticky="w", pady=15)

        ttk.Label(bounding_box, text="Duration", font=("Helvetica", 12)).grid(row=11, column=3, sticky="w")
        self.duration_label = ttk.Label(bounding_box, text="", font=("Helvetica", 12))
        self.duration_label.grid(row=11, column=4, sticky="w")

        self.end_time_button = ttk.Button(bounding_box, text="End  Time    ", command=self.stop_detection, style="Red.TButton")
        self.end_time_button.grid(row=11, column=5, columnspan=1, pady=15, sticky="ew")

        self.end_time_label = ttk.Label(bounding_box, text="", font=("Helvetica", 12, "bold"), foreground="red")
        self.end_time_label.grid(row=11, column=6, sticky="w")

        bounding_box.columnconfigure(1, weight=1)
        bounding_box.columnconfigure(4, weight=1)
        bounding_box.columnconfigure(7, weight=1)

        # Update the count label periodically
        self.update_count()

        # Add the refresh button at the bottom
        self.refresh_button = ttk.Button(self, text="Refresh", command=self.refresh, style="Green.TButton")
        self.refresh_button.grid(row=2, column=0, columnspan=8, pady=10, sticky="ew")

    def update_count(self):
        self.count.config(text=self.shared_count.get())
        self.after(1000, self.update_count)

    def start_detection(self):
        global start_time
        self.duration_label.config(text="")
        self.end_time_label.config(text="")
        self.shared_count.set("None")
        start_time = datetime.now()
        self.start_time_label.config(text=start_time.strftime("%H:%M:%S"))
        select_product = self.select_product.get()  # Get the selected product from the combobox
        stop_event.clear()  # Clear the stop event before starting the thread
        self.process = threading.Thread(target=video_processing_thread, args=(source, model, threshold, text_size, select_product, self.shared_count))
        self.process.start()

    def stop_detection(self):
        global end_time, duration
        if self.process:
            stop_event.set()  # Set the stop event to signal the thread to stop
            self.process.join(timeout=5)  # Wait for the thread to finish with a timeout
            self.process = None  # Reset the process thread
            end_time = datetime.now()
            self.end_time_label.config(text=end_time.strftime("%H:%M:%S"))
            duration = end_time - start_time
            self.duration_label.config(text=str(duration))
            # Call the method to create the report
            self.create_report(
                item_code=self.item_code.get(),
                product_name=self.select_product.get(),
                demand_value=demand_value,
                produced=self.shared_count.get(),
                temperature=current_temp_value,
                total_baskets=baskets_value,
                line_number=3,
                num_staff=operators_value,
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )

    def create_report(self, item_code, product_name, demand_value, produced, temperature, total_baskets, line_number, num_staff, start_time, end_time, duration):
        # Calculate the total man hour and average preparation time
        try:
            produced = int(produced) if produced != "None" else 0
            total_man_hour = int(num_staff) * duration.total_seconds() / 3600  # Convert duration to hours
            avg_preparation_time = duration.total_seconds() / produced if produced > 0 else "-"
        except ZeroDivisionError:
            avg_preparation_time = "-"

        # Create the report content
        report_content = f"""
        Production Summary

        Product ID: {item_code}
        Product Name: {product_name}
        Demand: {demand_value}
        Produced: {produced}
        Temperature: {temperature}
        Total Baskets: {total_baskets}
        Line Number: {line_number}
        Number of Staff: {num_staff}
        Total Manhour: {total_man_hour:.2f}
        Avg. Preparation Time: {avg_preparation_time if isinstance(avg_preparation_time, str) else f"{avg_preparation_time:.2f}"} seconds

        Time Start: {start_time.strftime("%H:%M:%S")}
        Time End: {end_time.strftime("%H:%M:%S")}
        Duration: {duration}
        """

        # Create the report directory with the current date if it doesn't exist
        todays_date = datetime.now().strftime("%d%b%Y").upper()
        report_dir = f"report/Date_{todays_date}"
        os.makedirs(report_dir, exist_ok=True)

        # Save the report to a file
        report_filename = f"{report_dir}/Report_{item_code}_{product_name.split(':')[1]}_{line_number}_{start_time.strftime('%H-%M-%S')}_to_{end_time.strftime('%H-%M-%S')}.txt"
        with open(report_filename, "w") as file:
            file.write(report_content)

        # Display the report in a popup window
        def on_close():
            popup.destroy()

        popup = tk.Toplevel()
        popup.title("Production Report")
        report_label = tk.Label(popup, text=report_content, justify=tk.LEFT, font=("Helvetica", 10))
        report_label.pack(padx=20, pady=20)
        ok_button = tk.Button(popup, text="OK", command=on_close, font=("Helvetica", 12))
        ok_button.pack(pady=10)

    def refresh(self):
        # Shutdown the running process if any
        if hasattr(self, 'process') and self.process and self.process.is_alive():
            stop_event.set()
            self.process.join(timeout=5)
        # Reload the configuration and input values
        self.shared_count.set("None")
        load_config()
        load_input_values()
        # Recreate the widgets
        for widget in self.winfo_children():
            widget.destroy()
        self.create_widgets()

    def update_time(self):
        current_time = datetime.now().strftime("%d %B %Y  %A  %H:%M:%S")
        self.time_label.config(text=f"Date Time:    {current_time}    ")
        self.after(1000, self.update_time)  # Update every second

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = KitchenDemandApp()
    app.mainloop()
