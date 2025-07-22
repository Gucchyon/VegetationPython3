# Vegetation Analysis Web App

This web application analyzes vegetation in images using various vegetation indices.

## Features

- Upload and process images
- Calculate vegetation coverage
- Multiple vegetation indices (ExG, GRVI, etc.)
- Binary mask generation
- Batch processing capability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vegetation-analysis.git
cd vegetation-analysis
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

## Usage

1. Select thresholding method (Otsu or Manual)
2. Upload an image
3. View results and analysis

## License

MIT

This project uses [Streamlit](https://streamlit.io/), which is distributed under the MIT License:

Copyright (c) Streamlit Inc. (2018-2024)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

### Third-party Libraries and Licenses

- This project uses [opencv-python-headless](https://pypi.org/project/opencv-python-headless/), which is distributed under the MIT License. OpenCV itself is available under the Apache 2.0 License. See: https://github.com/opencv/opencv-python/blob/main/LICENSE.txt

- This project uses [NumPy](https://numpy.org/), which is distributed under the 3-Clause BSD License. Copyright (c) 2005-2025, NumPy Developers. See: https://numpy.org/doc/stable/license.html

- This project uses [pandas](https://pandas.pydata.org/), which is distributed under the 3-Clause BSD License. See: https://github.com/pandas-dev/pandas/blob/main/LICENSE

- This project uses [Pillow](https://python-pillow.org/), which is distributed under the open source MIT License. See: https://pillow.readthedocs.io/en/stable/about.html#license

- This project runs on [Python](https://www.python.org/), which is distributed under the Python Software Foundation License Version 2. See: https://docs.python.org/3/license.html

## Author

Your Name"# VegetationPython3" 
