# Feature Based Image Alignment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18laBUuI6MA6FloLEG1wZvG2JRhWqPRaC?usp=sharing)

## How to Run This App

### 1. Via Cloning This Repo

The first step is clone the repository by command: `git clone https://github.com/Akin01/Nodeflux-Intern-Project.git`

#### Running Inference server

- Change directory at project week 2: `cd week 2` and go to `Inference Server` directory: `cd "Feature Based Image Alignment/Inference Server"`
- Install dependencies using `pip` by command: `pip install -r requirements.txt`
- If you're using `pipenv` as package manage, run: `pipenv install` and activate your environment: `pipenv shell`
- Run the server using: `python app.py`

#### Running Streamlit Interface

- Change directory at project week 2: `cd week 2` and go to `streamlit interface` directory: `cd "Feature Based Image Alignment/streamlit interface"`
- Install dependencies using `pip` by command: `pip install -r requirements.txt`
- If you're using `pipenv` as package manage, run: `pipenv install` and activate your environment: `pipenv shell`
- Run the server using: `python -m streamlit run app.py`

### 2. Via Docker

You can build your own image by Dockerfile.

#### Build Inference Server Image

- Change directory at project week 2: `cd week 2` and go to `Inference Server` directory: `cd "Feature Based Image Alignment/Inference Server"`
- Build your own image by command: `docker build -t "[your_image_name]:latest" .` (without square bracket)
- Run docker container locally via: `docker run -p 7001:7001 [your_image_name]` (without square bracket)
- Your server will be running at `http://127.0.0.1:7001`.
- Inference endpoint: `/images/process`

#### Build Streamlit Interface Image

- Change directory at project week 2: `cd week 2` and go to `streamlit interface` directory: `cd "Feature Based Image Alignment/streamlit interface"`
- Build your own image by command: `docker build -t "[your_image_name]:latest" .` (without square bracket)
- Run docker container locally via: `docker run -p 8501:8501 [your_image_name]` (without square bracket)
- To see your application, open `http://127.0.0.1:8501` at your browser.

If all the process succesfully done, you will see :

#### mock image

![mock_image](https://raw.githubusercontent.com/Akin01/Nodeflux-Intern-Project/master/Week%201/assets/demo-upload-1.png)

#### unalignment image

![unalign_image](https://raw.githubusercontent.com/Akin01/Nodeflux-Intern-Project/master/Week%201/assets/demo-upload-2.png)

#### alignment result

![alignment_image](https://raw.githubusercontent.com/Akin01/Nodeflux-Intern-Project/master/Week%201/assets/demo-result.png)
