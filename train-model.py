import argparse
from pathlib import Path
from typing import Callable, Tuple
import dask.array as da
import numpy as np
import time
from dasf.transforms.base import Transform
from enum import Enum

import matplotlib.pyplot as plt

from dasf_seismic.attributes.complex_trace import Envelope, CosineInstantaneousPhase,InstantaneousFrequency
from dasf.transforms import PersistDaskData
from dasf.pipeline import Pipeline
from dasf.datasets import Dataset
from dasf.ml.xgboost import XGBRegressor
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler

class SaveModel(Transform):
    def __transform_generic(self, model):
        model.save_model()

    def _lazy_transform_cpu(self, X=None, **kwargs):
        return self.__transform_generic(X)

    def _transform_cpu(self, X=None, **kwargs):
        dfs = self.__transform_generic(X, y)

        if is_array(dfs) and not is_dask_array(dfs):
            datas = np.stack(dfs, axis=-1)
            datas = pd.DataFrame(datas, columns=y)
        else:
            datas = dfs

        return datas

class ArraysToDataFrame(Transform):
    def __init__(self, inlineWindow, traceWindow, sampleWindow):
        self.inlineWindow = inlineWindow
        self.sampleWindow = sampleWindow
        self.traceWindow = traceWindow

    def __create_dataframe_with_neighbors(self, data):
        print()
        print()
        print()
        print()
        print()
        print(type(data))
        print(data)
        print()
        print()
        print()
        print()
        # Get the dimensions of the input array
        a, b, c = data.shape

        # Create empty lists to store the data for the DataFrame
        neighbors = []

        # Iterate over each point in the input array
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    # Store the point coordinates

                    # Initialize lists to store the neighbors' values
                    x_vals = []
                    y_vals = []
                    z_vals = []

                    # Iterate over the neighbors' indices in each axis
                    for x_offset in range(-self.inlineWindow, self.inlineWindow+1):
                      if x_offset == 0:
                        continue

                      x_idx = i + x_offset
                      if x_idx < 0 or x_idx >= a:
                        x_idx = 0 if x_idx < 0 else a - 1

                      x_vals.append(data[x_idx, j, k])

                    for y_offset in range(-self.traceWindow, self.traceWindow+1):
                      if y_offset == 0:
                        continue

                      y_idx = j + y_offset
                      if y_idx < 0 or y_idx >= b:
                        y_idx = 0 if y_idx < 0 else b - 1

                      y_vals.append(data[i, y_idx, k])

                    for z_offset in range(-self.sampleWindow, self.sampleWindow+1):
                      if z_offset == 0:
                        continue

                      z_idx = k + z_offset
                      if z_idx < 0 or z_idx >= c:
                        z_idx = 0 if z_idx < 0 else c - 1

                      z_vals.append(data[i, j, z_idx])

                    # Combine the neighbor values into a single list
                    neighbor_values = [data[i, j, k]] + x_vals + y_vals + z_vals


                    # Store the neighbor values
                    neighbors.append(neighbor_values)

        df = pd.DataFrame(neighbors)

        return df

    def __transform_generic(self, X):
        dfs = self.__create_dataframe_with_neighbors(X)
        return dfs

    def _lazy_transform_cpu(self, X=None, **kwargs):
        return self.__transform_generic(X)

    def _transform_cpu(self, X=None, **kwargs):
        dfs = self.__transform_generic(X, y)

        if is_array(dfs) and not is_dask_array(dfs):
            datas = np.stack(dfs, axis=-1)
            datas = pd.DataFrame(datas, columns=y)
        else:
            datas = dfs

        return datas

class Attributes(Enum):
    ENVELOPE = "ENVELOPE"
    INST_FREQ = "INST-FREQ"
    COS_INST_PHASE = "COS-INST-PHASE"

class MyDataset(Dataset):
    """Classe para carregar dados de um arquivo .npy
    """
    def __init__(self, name: str, data_path: str, chunks: str = "32Mb"):
        """Instancia um objeto da classe MyDataset

        Parameters
        ----------
        name : str
            Nome simbolicamente associado ao dataset
        data_path : str
            Caminho para o arquivo .npy
        chunks: str
            Tamanho dos chunks para o dask.array
        """
        super().__init__(name=name)
        self.data_path = data_path
        self.chunks = chunks
        
    def _lazy_load_cpu(self):
        return da.from_zarr(self.data_path, chunks=self.chunks)
    
    def _load_cpu(self):
        return np.load(self.data_path)
    
    @task_handler
    def load(self):
        ...

def create_executor(address: str=None) -> DaskPipelineExecutor:
    """Cria um DASK executor

    Parameters
    ----------
    address : str, optional
        Endereço do Scheduler, by default None

    Returns
    -------
    DaskPipelineExecutor
        Um executor Dask
    """
    if address is not None:
        addr = str(address.split(":")[0])
        port = str(address.split(":")[-1])
        print(f"Criando executor. Endereço: {addr}, porta: {port}")
        return DaskPipelineExecutor(local=False, use_gpu=False, address=addr, port=port)
    else:
        return DaskPipelineExecutor(local=True, use_gpu=False)
        
def create_pipeline(dataset_path: str, executor: DaskPipelineExecutor, attribute:str = None, inlineWindow:int = None, traceWindow:int = None, sampleWindow:int = None) -> Tuple[Pipeline, Callable]:
    """Cria o pipeline DASF para ser executado

    Parameters
    ----------
    dataset_path : str
        Caminho para o arquivo .npy
    executor : DaskPipelineExecutor
        Executor Dask

    Returns
    -------
    Tuple[Pipeline, Callable]
        Uma tupla, onde o primeiro elemento é o pipeline e o segundo é último operador (kmeans.fit_predict), 
        de onde os resultados serão obtidos.
    """
    print("Criando pipeline....")
    # Declarando os operadores necessários
    dataset = MyDataset(name="F3 dataset", data_path=dataset_path)
    envelope = Envelope()
    cosPhase = CosineInstantaneousPhase()
    instFreq = InstantaneousFrequency()
    data2df = ArraysToDataFrame(inlineWindow, traceWindow, sampleWindow)
    # Persist é super importante! Se não cada partial_fit do k-means vai computar o grafo até o momento!
    # Usando persist, garantimos que a computação até aqui já foi feita e está em memória distribuida.
    persist = PersistDaskData()
    y = None

    # Cria um objeto XGBoost
    boostedTree = XGBRegressor()
    # Compondo o pipeline
    pipeline = Pipeline(
        name="F3 seismic attributes",
        executor=executor
    )
    pipeline.add(dataset)

    if attribute == Attributes.ENVELOPE.value:
        pipeline.add(envelope, X=dataset)
        pipeline.add(data2df, X=dataset)
    elif attribute == Attributes.INST_FREQ.value:
        pipeline.add(instFreq, X=dataset)
        pipeline.add(data2df, X=dataset)
    elif attribute == Attributes.COS_INST_PHASE.value:
        pipeline.add(cosPhase, X=dataset)
        pipeline.add(data2df, X=dataset)
    else:
        print("\033[33m Atributo não reconhecido \033[0m")
        return

    pipeline.add(persist, X=data2df)
    pipeline.add(boostedTree.fit, X=persist,  y=envelope)
    pipeline.add(save_model,boostedTree)
    pipeline.visualize(filename="./train-pipeline.jpg")
    
    # Retorna o pipeline e o operador kmeans, donde os resultados serão obtidos
    return pipeline, boostedTree

def run(pipeline: Pipeline, last_node: Callable, output: str = None) -> np.ndarray:
    """Executa o pipeline e retorna o resultado

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline a ser executado
    last_node : Callable
        Último operador do pipeline, de onde os resultados serão obtidos

    Returns
    -------
    np.ndarray
        NumPy array com os resultados
    """
    print("Executando pipeline")
    start = time.time()
    pipeline.run()
    res = pipeline.get_result_from(last_node)
    res = res.compute()
    end = time.time()


    if output:
        print(f"\033[31m Salvando o modelo treinado em {output} \033[0m")
        last_node.save_model(model-filename.json)
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--attribute", type=str, required=True, help="Atributo que será usado para treinar o modelo\
                                                                        \n Valore possíveis são:\n\
                                                                        ENVELOPE: Envelope|\n\
                                                                        INST-FREQ: Frequência instantânea\n\
                                                                        COS-INST-PHASE: Cosseno instantâneo da fase")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o arquivo com dados de entrada .zarr")
    parser.add_argument("--samples-window", type=int, required=True, help="Número de vizinhos na dimensão das amostras de um traço")
    parser.add_argument("--trace-window", type=int, required=True, help="Número de vizinhos na dimensão dos traços de uma inline")
    parser.add_argument("--inline-window", type=int, required=True, help="Número de vizinhos na dimensão das inlines")
    parser.add_argument("--address", type=str, default=None, help="Endereço do scheduler. Formato: HOST:PORT")
    parser.add_argument("--output", type=str, default=None, help="Local para salvar o modelo treinado com extensão .json")
    args = parser.parse_args()
   
    # Criamos o executor
    executor = create_executor(args.address)
    # Depois o pipeline
    pipeline, last_node = create_pipeline(args.data, executor, args.attribute, args.samples_window, args.trace_window, args.inline_window)
    # Executamos e pegamos o resultado
    res = run(pipeline, last_node, args.output)
    print(f"O resultado é um array com o shape: {res.shape}")
    
    # Podemos fazer o reshape e printar a primeira inline
    if args.save_inline_fig is not None:
        res = res.reshape((401, 701, 255))
        import matplotlib.pyplot as plt
        plt.imsave(args.save_inline_fig, res[0], cmap="viridis")
        print(f"Figura da inline 0 salva em {args.save_inline_fig}")
    
