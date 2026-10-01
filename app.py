import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import MicroCNN
from world import make_world


# PAGE SETUP

st.set_page_config(
    page_title="Micro-VLA Spatial Reasoner",
    page_icon="🎯",
    layout="wide"
)

# CUSTOM CSS


st.markdown(
    """
    <style>

    /* Reduce empty space at the top */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Center the main heading */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    /* Center subtitle */
    .subtitle {
        text-align: center;
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1rem;
    }

    /* Prediction heading */
    .prediction-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
        margin-bottom: 0.7rem;
    }

    /* Welcome / status message */
    .status-box {
        padding: 0.55rem 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 0.8rem;
        font-weight: 600;
        background: rgba(70, 130, 180, 0.12);
    }

    /* Make buttons slightly more compact */
    div.stButton > button {
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# LOAD MODEL

@st.cache_resource
def load_model():

    model = MicroCNN()

    model.load_state_dict(
        torch.load(
            "microcnn.pth",
            weights_only=True
        )
    )

    model.eval()

    return model


model = load_model()


# COLOURS

colour_names = ["Red", "Green", "Blue"]
colour_emojis = ["🔴", "🟢", "🔵"]



# SESSION STATE


if "image" not in st.session_state:

    image, target = make_world()

    st.session_state.image = image
    st.session_state.target = target


if "prediction_result" not in st.session_state:

    st.session_state.prediction_result = None


image = st.session_state.image
target = st.session_state.target


# HEADER

st.markdown(
    '<div class="main-title">🎯 Micro-VLA Spatial Reasoner</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        A tiny PyTorch CNN that learns where coloured objects
        are located in a synthetic image.
    </div>
    """,
    unsafe_allow_html=True
)


# PREDICTION INFORMATION

prediction = st.session_state.prediction_result


if prediction is None:

    st.markdown(
        """
        <div class="status-box">
            🌍 Welcome! Choose a coloured object and let the CNN find it.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    colour_index = prediction["colour_index"]

    st.markdown(
        f"""
        <div class="prediction-title">
            {colour_emojis[colour_index]} {colour_names[colour_index]} — Prediction
        </div>
        """,
        unsafe_allow_html=True
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Actual Centre",
            f"({prediction['actual_x']:.1f}, "
            f"{prediction['actual_y']:.1f})"
        )

    with metric2:

        st.metric(
            "CNN Prediction",
            f"({prediction['predicted_x']:.1f}, "
            f"{prediction['predicted_y']:.1f})"
        )

    with metric3:

        st.metric(
            "Pixel Error",
            f"{prediction['error']:.2f} px"
        )



# MAIN TWO-COLUMN LAYOUT

left, right = st.columns(
    [0.9,1.5],
    gap="medium"
)


# LEFT SIDE — CONTROLS


with left:

    st.subheader("🎨 Find an Object")

    st.write(
        "Choose which coloured block you want "
        "the CNN to locate."
    )

    selected_colour = st.radio(
        "Target colour",
        colour_names
    )

    st.write("")

    find_object = st.button(
        "🚀 FIND OBJECT",
        use_container_width=True
    )

    new_world = st.button(
        "🌍 Generate New World",
        use_container_width=True
    )



# NEW WORLD

if new_world:

    image, target = make_world()

    st.session_state.image = image
    st.session_state.target = target

    # Remove old prediction
    st.session_state.prediction_result = None

    st.rerun()



# FIND OBJECT

if find_object:

    colour_index = colour_names.index(
        selected_colour
    )

    # Add batch dimension
    image_batch = image.unsqueeze(0)

    # CNN prediction
    with torch.no_grad():

        prediction_tensor = model(
            image_batch
        )[0]

    # Actual position
    actual_position = target[colour_index]

    # Predicted position
    predicted_position = prediction_tensor[
        colour_index
    ]

    # CONVERT NORMALIZED COORDINATES TO PIXELS


    actual_x = (
        actual_position[0].item()
        * 31
    )

    actual_y = (
        actual_position[1].item()
        * 31
    )

    predicted_x = (
        predicted_position[0].item()
        * 31
    )

    predicted_y = (
        predicted_position[1].item()
        * 31
    )


    # PIXEL ERROR

    error = (
        (predicted_x - actual_x) ** 2
        +
        (predicted_y - actual_y) ** 2
    ) ** 0.5


    
    # SAVE RESULT
    

    st.session_state.prediction_result = {

        "colour_index": colour_index,

        "actual_x": actual_x,
        "actual_y": actual_y,

        "predicted_x": predicted_x,
        "predicted_y": predicted_y,

        "error": error
    }

    st.rerun()



# RIGHT SIDE — WORLD


with right:

    if prediction is None:

        st.subheader("🌍 Current World")

        image_display = (
            image
            .permute(1, 2, 0)
            .numpy()
        )

        fig, ax = plt.subplots(
            figsize=(3,3)
        )

        ax.imshow(
            image_display,
            interpolation="nearest"
        )

        ax.set_xlim(-1, 32)
        ax.set_ylim(32, -1)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            "Synthetic 32 × 32 RGB World",
            fontsize=12,
            fontweight="bold"
        )

        st.pyplot(
            fig,
            width=430
        )

        plt.close(fig)


    else:

        colour_index = prediction[
            "colour_index"
        ]

        st.subheader(
            f"{colour_emojis[colour_index]} "
            f"{colour_names[colour_index]} — Predicted World"
        )

        image_display = (
            image
            .permute(1, 2, 0)
            .numpy()
        )

        fig, ax = plt.subplots(
            figsize=(3,3)
        )

        ax.imshow(
            image_display,
            interpolation="nearest"
        )


       
        # ACTUAL CENTRE
        

        ax.scatter(
            prediction["actual_x"],
            prediction["actual_y"],
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="white",
            linewidths=2,
            label="Actual"
        )


        # CNN PREDICTION
        

        ax.scatter(
            prediction["predicted_x"],
            prediction["predicted_y"],
            marker="+",
            s=120,
            linewidths=2.5,
            label="CNN"
        )


       
        # ERROR LINE


        ax.plot(
            [
                prediction["actual_x"],
                prediction["predicted_x"]
            ],
            [
                prediction["actual_y"],
                prediction["predicted_y"]
            ],
            "--",
            linewidth=1.5,
            alpha=0.8
        )


        ax.set_xlim(-1, 32)
        ax.set_ylim(32, -1)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            "CNN Location Prediction",
            fontsize=12,
            fontweight="bold"
        )

        ax.legend(
            fontsize=8,
            loc="upper right",
            framealpha=0.8,
            borderpad=0.4,
            handlelength=1.2,
            handletextpad=0.4
        )

        st.pyplot(
            fig,
            width=430
        )

        plt.close(fig)