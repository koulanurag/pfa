# Policy Fusion Architecture - Pytorch

## Installation
 * Install [Pytorch](http://pytorch.org/) [Version : 0.3.1] - (Platform dependent)
 * To install other requirements:
    ```sh
        pip install -r requirements.txt
    ```

## Usage
  * If using render option: Start the visdom server before running the script
    ```sh
    python -m visdom.server
    ```
  * Policy Gradient:
      * Training: ```python main_pg.py --train```
      * Testing: ```python main_pg.py --test```
  * Hybrid Policy Gradient:
      * Training: ```python main_hpg.py --train```
      * Testing: ```python main_hpg.py --test```
