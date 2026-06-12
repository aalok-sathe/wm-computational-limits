# Tutorials

## Config structure

```mermaid
flowchart TD
    A(<b>experiment</b> fa:fa-microscope) 
    A --> | | Indep( <b>independent variables<b> fa:fa-gears\n<i>specified as a list</i>\n<i></i>
    )
    A --> | | Dep( <b>conditional variables</b>\nlookup table fa:fa-square-root-variable)
    
    Indep --> | | I1[ <b>Variable X</b> 
    <code>typing.List:1..N</code> ]
    Indep --> | | C[co-varying variable lists specified together; must be matched in length]
    C --o C1
    C --o C2 
    C1[ <b>Variable Y</b> 
    <code>typing.List:1..M</code> ]
    C2[ <b>Variable Z</b>
    <code>typing.List:1..M</code> ]
    C1 o--o|co-varies with|C2

    Dep --> | | D1[ condition-set D1 ]
    K1[<b>key: conditions to match</b>
    X='lstm'; N_back=4]
    V1[<b>variables to apply</b>
        1. trainer.learning rate: 1e-4
        2. model.n_layers: 2]
    D1 --o K1 
    K1 --> V1

    Dep --> | | D2[ condition-set D2 ]
    K2[<b>key: conditions to match</b>
    X='lstm'; Ref_back=3]
    V2[<b>variables to apply</b>
        1. trainer.learning rate: 2e-3
        2. model.n_layers: 2
        3. model.d_model: 256]
    D2 --o K2 
    K2 --> V2

    Dep --> | | D3[ condition-set D.... ]

    style A fill:#FFCCFB,stroke:#000,stroke-width:4px,color:#000000

    style Indep fill:#CCFFD0,stroke:#000,stroke-width:4px,color:#000000
    style Dep fill:#C1CABF,stroke:#000,stroke-width:4px,color:#000000

    style C1 fill:#21FF2124,stroke:#F6A981,stroke-width:3px,color:#ff
    style C2 fill:#21FF2124,stroke:#F6A981,stroke-width:3px,color:#ff
    style I1 fill:#21FF2124,stroke:#000,stroke-width:3px,color:#ff

    style D1 fill:#217AFF24,stroke:#000,stroke-width:3px,color:#ff
    style D2 fill:#217AFF24,stroke:#000,stroke-width:3px,color:#ff
    style D3 fill:#217AFF24,stroke:#000,stroke-width:3px,color:#ff

```


## Create your first experiment sweep
1. identify manipulations of interest (see what variations the library already supports using `python -m workingmem -h`).
2. write/modify a config defining experimental conditions ([example](./configs/sample_conditional_config))
    - configs follow an "independent variables" and "conditional variables" format---independent variables are enumerated as lists
      that yield a cross-product over all possible combinations of their variation. "conditional variables" are typically hyperparameters
      that need to be looked up dependent on the particular condition.  
3. use `python -m workingmem --wandb.create_sweep` along with the flag `--wandb.from_config [path/to/config]` to define individual experimental conditions.
  at this point, the library evaluates a cross-product over all possible conditions in your experiment and creates individual
  W&B "sweeps" for each condition. this enables separate tracking of the progress of experiments in a web browser, as well as unique-ID-based
  retrieval after the experiment finishes for clean, reproducible science.
