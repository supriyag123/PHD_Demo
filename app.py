{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/supriyag123/PHD_Demo/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# app.py\n",
        "\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import datetime as dt\n",
        "import time\n",
        "\n",
        "# --- Settings ---\n",
        "SENSOR_COUNT = 5\n",
        "THRESHOLD = 70\n",
        "\n",
        "# --- Generate Dummy Data ---\n",
        "def create_dummy_df(n=200, anomaly_prob=0.05, seed=42):\n",
        "    np.random.seed(seed)\n",
        "    ts = [dt.datetime.now() + dt.timedelta(seconds=i) for i in range(n)]\n",
        "    data = {}\n",
        "    for i in range(1, SENSOR_COUNT + 1):\n",
        "        base = np.random.normal(50, 5, n)\n",
        "        mask = np.random.rand(n) < anomaly_prob\n",
        "        base[mask] += np.random.normal(20, 5, mask.sum())  # anomalies\n",
        "        data[f\"sensor_{i}\"] = base\n",
        "    df = pd.DataFrame(data)\n",
        "    df.insert(0, \"timestamp\", ts)\n",
        "    return df\n",
        "\n",
        "# --- UI Setup ---\n",
        "st.set_page_config(layout=\"wide\")\n",
        "st.title(\"🛰️ Real-Time Industrial IoT Sensor Dashboard\")\n",
        "st.markdown(\"\"\"\n",
        "Welcome to the **Simulated IoT Monitoring App**.\n",
        "\n",
        "🔹 Simulates real-time sensor readings\n",
        "🔹 Displays streaming line charts & anomalies\n",
        "🔹 Shows agent-style reasoning for detected events\n",
        "\"\"\")\n",
        "\n",
        "# --- Sidebar Controls ---\n",
        "st.sidebar.title(\"⚙️ Controls\")\n",
        "speed = st.sidebar.slider(\"⏩ Playback speed\", 0.1, 5.0, 1.0)\n",
        "window = st.sidebar.slider(\"🕓 Window size (sec)\", 5, 60, 20)\n",
        "st.sidebar.markdown(\"---\")\n",
        "st.sidebar.markdown(\"👤 Built by [Your Name](https://example.com)\")\n",
        "st.sidebar.image(\"https://upload.wikimedia.org/wikipedia/commons/4/44/Industrial_icon.png\", width=150)\n",
        "\n",
        "# --- Load Data ---\n",
        "df = create_dummy_df()\n",
        "sensor_cols = df.columns[1:]\n",
        "selected_sensors = st.multiselect(\"📊 Select sensors to display\", sensor_cols.tolist(), default=sensor_cols[:3])\n",
        "\n",
        "# --- App Layout ---\n",
        "alert = st.empty()\n",
        "chart = st.empty()\n",
        "metrics = st.columns(len(selected_sensors))\n",
        "agent_box = st.container()\n",
        "\n",
        "# --- Streaming Loop ---\n",
        "for i in range(len(df)):\n",
        "    now = df['timestamp'].iloc[i]\n",
        "    window_df = df[df['timestamp'] >= now - pd.Timedelta(seconds=window)]\n",
        "\n",
        "    # --- Chart Update ---\n",
        "    with chart.container():\n",
        "        st.line_chart(window_df.set_index(\"timestamp\")[selected_sensors])\n",
        "\n",
        "    # --- Real-Time Metrics ---\n",
        "    for j, sensor in enumerate(selected_sensors):\n",
        "        latest_val = window_df[sensor].iloc[-1]\n",
        "        metrics[j].metric(sensor, round(latest_val, 2))\n",
        "\n",
        "    # --- Anomaly Check ---\n",
        "    anomalies = (window_df[selected_sensors] > THRESHOLD).any(axis=1)\n",
        "    if anomalies.any():\n",
        "        alert.warning(\"⚠️ Anomaly detected in the selected window!\")\n",
        "        agent_message = \"Agent: Sudden spike detected. Recommend checking affected sensors.\"\n",
        "    else:\n",
        "        alert.success(\"✅ System operating normally.\")\n",
        "        agent_message = \"Agent: All sensor values within expected range.\"\n",
        "\n",
        "    # --- Agent Reasoning Panel ---\n",
        "    with agent_box:\n",
        "        st.markdown(\"### 🤖 Agent Reasoning\")\n",
        "        st.info(agent_message)\n",
        "\n",
        "    time.sleep(1.0 / speed)"
      ],
      "metadata": {
        "id": "oOHjAhnZs1Az",
        "outputId": "666af7b6-dafc-441c-9fc1-ce4545913a02",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting streamlit_app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "W-RMQkgLak9I"
      }
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "0zEUESMBM_Eo"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bQ6ZgIvrdRNp"
      },
      "source": [
        "# New Section"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "machine_shape": "hm",
      "mount_file_id": "https://github.com/supriyag123/PHD_Pub/blob/main/PAPER_GDRNet_METROPM_Step1.ipynb",
      "authorship_tag": "ABX9TyPwwAy9zhileYD7m11zV9FT",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}