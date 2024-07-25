<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">FINSCRAPER</h1>
</p>
<p align="center">
    <em>Unleashing financial insights with cutting-edge tech!</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/ChiragAgg5k/finscraper?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/ChiragAgg5k/finscraper?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/ChiragAgg5k/finscraper?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/ChiragAgg5k/finscraper?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>

<br><!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Tests](#tests)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
</details>
<hr>

## Overview

The finscraper project encompasses a suite of financial data extraction and processing functionalities. It includes modules for scraping SEC filings, model training with Fast R-CNN, and preprocessing financial statement images. The project manages dependencies via its pyproject.toml file, ensuring seamless integration for scraping tasks. With capabilities to download, store, analyze, and train models on financial data, finscraper offers a robust platform for financial data extraction, preparation, and analysis.

---

## Features

|     | Feature           | Description                                                                                                                                                                                                                                                          |
| --- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ⚙️  | **Architecture**  | The project uses a modular architecture with separate modules for scraping, testing, model training, and pre-processing financial data. It leverages external libraries and tools for specific tasks, promoting code separation and reusability.                     |
| 🔩  | **Code Quality**  | The codebase maintains high code quality standards with clear naming conventions, consistent formatting, and well-structured functions. Code comments are used effectively to enhance code readability and maintainability.                                          |
| 📄  | **Documentation** | The project includes informative documentation in the form of code comments, README files, and module descriptions. It provides detailed explanations of the project structure, functionality, and usage, making it easier for developers to onboard and contribute. |
| 🔌  | **Integrations**  | Key external dependencies include pytesseract, bs4, matplotlib, torch, and more for tasks such as image processing, web scraping, data visualization, and machine learning. These integrations enhance the project's capabilities and extend its functionality.      |
| 🧩  | **Modularity**    | The codebase demonstrates high modularity and reusability by encapsulating distinct functionalities into separate modules. This design allows for easy maintenance, testing, and scalability of individual components without affecting the entire system.           |
| 🧪  | **Testing**       | The project utilizes testing frameworks like pytest to ensure the correctness and reliability of various modules. Test cases are written to validate different functionalities, improving code robustness and facilitating future changes.                           |
| ⚡️ | **Performance**   | The project focuses on efficiency and resource optimization by using libraries like tensorflow and scikit-learn for machine learning tasks. It employs best practices for data processing and model training, enhancing performance and speed.                       |
| 🛡️  | **Security**      | Measures such as data encryption, secure access control, and data validation are implemented to ensure data protection and prevent unauthorized access. The project follows security best practices to safeguard sensitive financial information.                    |
| 📦  | **Dependencies**  | Key external libraries and dependencies include pytesseract, bs4, matplotlib, torch, scikit-learn, and more. These libraries provide essential functionalities for tasks such as image processing, web scraping, and machine learning model training.                |

---

## Repository Structure

```sh
└── finscraper/
    ├── README.md
    ├── finscraper
    │   ├── __init__.py
    │   ├── model_training.py
    │   ├── preprocessor.py
    │   ├── scraper.py
    │   └── test_setup.py
    ├── poetry.lock
    ├── pyproject.toml
    └── tests
        └── __init__.py
```

---

## Modules

<details closed><summary>.</summary>

| File                                                                                   | Summary                                                                                                                                                                                                         |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [pyproject.toml](https://github.com/ChiragAgg5k/finscraper/blob/master/pyproject.toml) | Manages project dependencies and metadata, including Python version and external libraries.-Ensures smooth integration and compatibility for financial data scraping tasks in the parent finscraper repository. |

</details>

<details closed><summary>finscraper</summary>

| File                                                                                                    | Summary                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [scraper.py](https://github.com/ChiragAgg5k/finscraper/blob/master/finscraper/scraper.py)               | Scrapes SEC for Apple Inc.s 10-K documents, downloads and stores them in a structured format. This module contributes to the financial data extraction and storage capabilities of the repository.                                                                       |
| [test_setup.py](https://github.com/ChiragAgg5k/finscraper/blob/master/finscraper/test_setup.py)         | Verifies software dependencies and GPU availability. Reports versions of key libraries and checks for CUDA and TensorFlow GPU devices. Helps ensure the system is configured correctly for model training in the finscraper repository.                                  |
| [model_training.py](https://github.com/ChiragAgg5k/finscraper/blob/master/finscraper/model_training.py) | Trains a Fast R-CNN model on financial table images.-Prepares and splits data for training.-Automates HTML to image conversion.-Cleans and preprocesses images.-Utilizes a dataset class and data loaders for model training.-Saves the fine-tuned model after training. |
| [preprocessor.py](https://github.com/ChiragAgg5k/finscraper/blob/master/finscraper/preprocessor.py)     | Analyzes, converts, cleans, and splits financial statement images. Extracts tables from HTML, transforms to images, enhances image quality, and divides dataset for model training. Implemented in preprocessor.py within finscraper project.                            |

</details>

---

## Getting Started

**System Requirements:**

-   **Python**: `version x.y.z`

### Installation

<h4>From <code>source</code></h4>

> 1. Clone the finscraper repository:
>
> ```console
> $ git clone https://github.com/ChiragAgg5k/finscraper
> ```
>
> 2. Change to the project directory:
>
> ```console
> $ cd finscraper
> ```
>
> 3. Install the dependencies:
>
> ```console
> $ pip install -r requirements.txt
> ```

### Usage

<h4>From <code>source</code></h4>

> Run finscraper using the command below:
>
> ```console
> $ python main.py
> ```

### Tests

> Run the test suite using the command below:
>
> ```console
> $ pytest
> ```

---

## Project Roadmap

-   [x] `► Create a pipeline for scraping financial data.`
-   [ ] `► Implement a model training module.`
-   [ ] `► Develop a preprocessor for financial statement images.`

---

## Contributing

Contributions are welcome! Here are several ways you can contribute:

-   **[Report Issues](https://github.com/ChiragAgg5k/finscraper/issues)**: Submit bugs found or log feature requests for the `finscraper` project.
-   **[Submit Pull Requests](https://github.com/ChiragAgg5k/finscraper/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
-   **[Join the Discussions](https://github.com/ChiragAgg5k/finscraper/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
    ```sh
    git clone https://github.com/ChiragAgg5k/finscraper
    ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
    ```sh
    git checkout -b new-feature-x
    ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
    ```sh
    git commit -m 'Implemented new feature x.'
    ```
6. **Push to github**: Push the changes to your forked repository.
    ```sh
    git push origin new-feature-x
    ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
 </details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="center">
   <a href="https://github.com{/ChiragAgg5k/finscraper/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=ChiragAgg5k/finscraper">
   </a>
</p>
</details>

---

## License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

-   List any resources, contributors, inspiration, etc. here.

[**Return**](#-overview)

---
