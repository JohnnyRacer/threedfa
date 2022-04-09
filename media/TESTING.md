# ThreeDFA Testing ⚙️ 🏗️
---
#### All images in the testing sample were gathered were licensed under the Unsplash license, feel free to use or modify as you like.
## The testing scripts are found within the `tests` folder in the repo's root directory. Below is a file tree of the tests and scripts that would be run.


```
tests/
├──utils.py
├── common/
│   ├── hash_verify.py
│   └── landmarks_verify.py
├── overlay/
│   ├── dense.py 
│   └── render.py
└── export/
    ├── uv_texture.py
    └── serialization.py
```

## Quick explanation and run down of the tests:

### *tests/utils.py*

---

Shared utilities that the testing scripts require.

### *tests/common/*

---

#### **hash_verify.py**

Checks the integrity of the installed threedfa module by verifying the hash checksum of the test results and downloaded required files. This is done to ensure both the files have been downloaded sucessfully and function verification via the output hashes.

#### **landmarks_verify.py** 

Checks the validity of the extracted facial landmarks against a pre-computed sample and ensures the bounding box is the same.

### *tests/overlay/*

---

#### **dense.py**

Tests if the derived dense landmarks are correct by comparing with known pre-computed samples.

#### **render.py**

Tests if the 3D rendering module has been built without errors and is working correctly. Generates the outputs from all the `overlay` function.

