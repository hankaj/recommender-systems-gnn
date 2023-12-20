# Graph Neural Networks in Recommender Systems

## How to Run the Project

Make sure you have Python and Rust installed. Follow these steps to run the recommender-systems-gnn project:

1. Navigate to the project directory:
    ```bash
    cd recommender-systems-gnn
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the collaborative filtering model remembering to set the desired parameters:
    ```bash
    python run_cf.py
    ```

4. Run the knowledge graph model remembering to set the desired parameters:
    ```bash
    python run_kg.py
    ```

5. To run multi-associative graph network, first change the directory and run the program then:
    ```
    cd witchnet/crates/magds
    cargo run --example movielens100k-recommend-items
    ```
