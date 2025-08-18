{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/supriyag123/PHD_Demo/blob/main/Streamlit.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "T-BJGPTX7XS5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fe6c1e7d-3ad6-483a-e070-c583deb978d1",
        "collapsed": true
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.11/dist-packages (1.48.0)\n",
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.11/dist-packages (7.3.0)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<7,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (8.2.1)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging<26,>=20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (25.0)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (11.3.0)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.29.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (4.14.1)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.11/dist-packages (from streamlit) (3.1.45)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.4.2)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.11/dist-packages (from pyngrok) (6.0.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.11/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (4.25.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.11/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2.0.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.11/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.4.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2025.8.3)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.11/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (25.3.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2025.4.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (0.36.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (0.26.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install streamlit pyngrok"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile streamlit_app.py\n",
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
        "\n",
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
        "    time.sleep(1.0 / speed)\n"
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
      "cell_type": "code",
      "source": [
        "!streamlit run streamlit_app.py &>/dev/null&"
      ],
      "metadata": {
        "id": "eC8pcMYYaUtE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import ngrok\n",
        "ngrok.kill()  # clean old tunnels\n",
        "\n",
        "# ⬇️ paste your token between the quotes\n",
        "ngrok.set_auth_token(\"30zHX6xv0Aqb6STXjMtLNsaa6el_45Xmu5rKv77UofrqjgkFR\")\n",
        "\n",
        "public_url = ngrok.connect(8501)  # tip: region option possible e.g. options={\"region\":\"au\"}\n",
        "print(\"Streamlit URL:\", public_url)"
      ],
      "metadata": {
        "id": "3K4QCcqas-qL",
        "outputId": "62065a54-d397-4cb4-97bf-b6010dc48d8e",
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
            "Streamlit URL: NgrokTunnel: \"https://d0f52aa92ddd.ngrok-free.app\" -> \"http://localhost:8501\"\n"
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
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "id": "Xp-Mv3Ksamxv"
      },
      "execution_count": null,
      "outputs": []
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
      "authorship_tag": "ABX9TyNJ56lD6Nkzb0fQDzl/l5Ax",
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