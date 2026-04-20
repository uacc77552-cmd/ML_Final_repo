import numpy as np
import gradio as gr
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_price(battery_power, blue, clock_speed, dual_sim, fc, four_g,
                  int_memory, m_dep, mobile_wt, n_cores, pc,
                  px_height, px_width, ram, sc_h, sc_w,
                  talk_time, three_g, touch_screen, wifi):

    
    features = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                          int_memory, m_dep, mobile_wt, n_cores, pc,
                          px_height, px_width, ram, sc_h, sc_w,
                          talk_time, three_g, touch_screen, wifi]])
    
    prediction = final_model.predict(features)[0]
    
    labels = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }
    
    return labels[prediction]

interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Battery Power"),
        gr.Radio([0,1], label="Bluetooth (0=No, 1=Yes)"),
        gr.Number(label="Clock Speed"),
        gr.Radio([0,1], label="Dual SIM"),
        gr.Number(label="Front Camera (MP)"),
        gr.Radio([0,1], label="4G"),
        gr.Number(label="Internal Memory"),
        gr.Number(label="Mobile Depth"),
        gr.Number(label="Weight"),
        gr.Number(label="Number of Cores"),
        gr.Number(label="Primary Camera"),
        gr.Number(label="Pixel Height"),
        gr.Number(label="Pixel Width"),
        gr.Number(label="RAM"),
        gr.Number(label="Screen Height"),
        gr.Number(label="Screen Width"),
        gr.Number(label="Talk Time"),
        gr.Radio([0,1], label="3G"),
        gr.Radio([0,1], label="Touch Screen"),
        gr.Radio([0,1], label="WiFi")
    ],
    outputs=gr.Textbox(label="Predicted Price Range"),
    title=" Mobile Price Classification ",
    description="predict price category"
)

interface.launch(share=True)

