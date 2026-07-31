# Red neuronal desde cero en C#

*[English](README.md) · **Español***

Una red neuronal *feed-forward* —o de propagación hacia delante— construida desde cero en C#. No
utiliza bibliotecas de machine learning ni frameworks especializados: solo arreglos y cálculo
numérico. Incluye propagación hacia atrás (`backpropagation`), operaciones aceleradas con SIMD,
serialización de modelos, comprobación de gradientes (`gradient check`) y benchmarks que respaldan
cada afirmación de rendimiento.

El repositorio incluye además una [**guía de estudio**](STUDY-GUIDE.es.md) que explica *por qué
existe cada pieza*: comienza por qué es una neurona, desarrolla la regla de la cadena y llega hasta
detalles de bajo nivel como las líneas de caché. También contiene dos ejemplos resueltos que puedes
seguir con una calculadora.

> **Convención de términos.** Esta versión evita traducciones literales que casi nadie usa en
> código, artículos académicos, documentación técnica o cursos. Por eso deja en inglés términos como
> `forward pass`, `mini-batch`, `learning rate`, `benchmark`, `dataset`, `overfitting`, `logits` y
> `gradient check`, explicándolos en español cuando hace falta.

```csharp
var net = new Sequential(inputs: 2)
    .Dense<Tanh>(4)
    .Dense<Sigmoid>(1)
    .Build(seed: 42);

net.Train(inputs, targets, epochs: 4000, learningRate: 0.5f);

float prediction = net.Predict([1f, 0f])[0];   // 0.9779

ModelIO.Save(net, "xor.nnm");
var loaded = ModelIO.Load("xor.nnm");          // predice de forma idéntica
```

## ¿Qué aporta este repositorio?

Hay muchos repositorios de «red neuronal desde cero». Este intenta aportar tres cosas menos comunes:

**Verifica su propio `backward pass`.** Una implementación de `backpropagation` con errores sutiles
puede continuar reduciendo la pérdida y parecer funcional durante un tiempo, lo que hace que estos
errores sean difíciles de encontrar. [`GradientCheck`](src/NN/GradientCheck.cs) compara cada
gradiente analítico con una estimación por diferencias finitas centradas. La batería de pruebas
comprueba la curva en U del error numérico que cabe esperar de un gradiente implementado
correctamente, e incluye una derivada incorrecta a propósito para demostrar que el `gradient check`
detecta efectivamente una derivada errónea.

**Mide sus propias afirmaciones y reporta la que resultó errónea.** Cada afirmación de rendimiento
de esta documentación tiene un [benchmark](bench/) detrás. Una de ellas —que una función de
activación implementada mediante un delegado sería significativamente más lenta que una genérica—
resultó ser **falsa**, y
[el informe lo dice](bench/README.es.md#3-activación-genérica-frente-a-delegado-la-afirmación-que-era-falsa)
en lugar de descartarla en silencio.

**Es honesta sobre sus límites.** [El §25 de la guía de estudio](STUDY-GUIDE.es.md) enumera lo que
esta implementación no hace y lo que costaría, en orden. Empieza por admitir que un *batched GEMM*
probablemente superaría cualquier optimización SIMD de este código, una afirmación que los
benchmarks ahora
[confirman parcialmente](bench/README.es.md#4-forwardbatch-un-resultado-nulo-deliberado).

## Cómo ejecutarlo

```bash
dotnet run --project src/NN.Demo      # perceptrón, XOR, guardar/cargar, gradient check, dos lunas
dotnet run -c Release --project src/NN.Mnist   # reconocimiento de dígitos manuscritos (~40 s)
dotnet test                           # la batería completa de pruebas
dotnet run -c Release --project bench/NN.Bench -- --filter '*'   # benchmarks
```

> **Sobre el idioma de la salida.** Los programas imprimen en inglés, así que los bloques de salida
> que aparecen abajo se reproducen tal cual: son lo que verás realmente en tu terminal. Traducirlos
> aquí sería describir un programa que no existe. El código, los nombres de identificadores y las
> banderas de línea de comandos se mantienen igual por la misma razón.

La aplicación de demostración ejecuta seis secciones. Las primeras cuatro utilizan ejemplos pequeños
basados en XOR: el perceptrón converge en AND, una capa oculta resuelve XOR, un modelo se guarda y
se carga de nuevo, y se ejecuta un `gradient check`:

```
Perceptron on AND: converged in 4 epochs

Network on XOR (2 -> 4 tanh -> 1 sigmoid):
  epoch  4000  loss 0.000350
  0 XOR 0 -> 0.0100    1 XOR 0 -> 0.9779
  0 XOR 1 -> 0.9826    1 XOR 1 -> 0.0226

Saved to xor.nnm (127 bytes), reloaded:
  round-trip is bit-for-bit identical

Gradient check: max relative error = 3.521E-004
```

Las dos últimas usan el conjunto de datos (`dataset`) *two moons* («dos lunas»), lo bastante grande
como para mostrar lo que XOR no puede mostrar: `mini-batches`, generalización a datos no vistos y
`overfitting`:

```
Two moons: 1500 noisy points, 1000 train / 500 test

Training (batch size 32, learning rate 0.3):
  epoch   train loss   train acc   test acc
      1       0.1511      83.9%     85.2%
    150       0.0177      98.1%     96.4%

Learned decision boundary (· = class 0, # = class 1):

  #####################################################·······
  ##################################################··········
  ###############################################·············
  #####################······###################··············
  ###################··········###############················
  #################··············############·················
  ################··················#######···················
  ##############··············································
  ############················································
  #########···················································

Overfitting: same problem, 20 training points, a much larger network
  4417 parameters, 20 training examples — 221x more knobs than data points

  epoch    train acc   test acc
   3000      100.0%     93.4%
```

> **Sobre los dígitos exactos.** La suma en coma flotante no es asociativa, y `Vector<float>` tiene
> ancho 4 en ARM pero 8 en AVX2. Eso hace que el producto escalar SIMD sume en distinto orden según
> la CPU. Los números citados aquí y en la guía de estudio provienen de un Apple M3 Pro con .NET 10.
> En otro hardware pueden moverse los últimos dígitos; las conclusiones deberían mantenerse. La CI
> se ejecuta deliberadamente en ambas arquitecturas.

## Leer dígitos manuscritos

Una demostración aparte entrena con [MNIST](src/NN.Mnist/): 60 000 dígitos manuscritos, 784 entradas
y 101 770 parámetros. Con dos capas densas, `backpropagation` y una salida `softmax` consigue
**98,0 % en dígitos que nunca vio, en 37 segundos**:

```
A training example (label 5):
              ==++**..**@@@@==
    ....--****@@@@@@@@@@%%**@@@@##::
  ..@@@@@@@@@@@@@@@@@@@@------::..
    %%@@@@@@@@@@####@@@@
          **@@--
            ##@@::
              --@@@@@@==
                    @@@@@@::
            ..++%%@@@@@@@@##
      ::%%@@@@@@@@##--
  **%%@@@@@@@@##--

  epoch   train loss   test acc      elapsed
      1      0.32598    93.53%        2.1s
     10      0.04203    97.86%       18.6s
     20      0.01349    98.02%       36.9s

Confusion matrix — rows are the true digit, columns the prediction:

             0     1     2     3     4     5     6     7     8     9    accuracy
    0      971     ·     1     1     ·     1     3     1     1     1      99.1%
    1        ·  1128     3     1     ·     1     1     1     ·     ·      99.4%
    5        3     1     ·    12     1   863     5     ·     5     2      96.7%
    9        2     2     1     7     4     3     2     4     2   982      97.3%
```

Después imprime los dígitos en los que se **equivocó**. Eso da una imagen más honesta del «98 %»
que el número por sí solo: la mayoría son lo bastante ambiguos como para que una persona también
dudara.

**El modelo entrenado se guarda, así que solo necesitas entrenarlo una vez.** Al volver a ejecutar
la demostración, carga 397 KB de pesos en lugar de reentrenar: **37 s → 4 ms**, con el mismo
98,02 %.

```bash
dotnet run -c Release --project src/NN.Mnist                     # reutiliza el modelo guardado
dotnet run -c Release --project src/NN.Mnist -- --predict 42     # clasifica una imagen de prueba
dotnet run -c Release --project src/NN.Mnist -- --image d.png    # clasifica tu propia imagen
dotnet run -c Release --project src/NN.Mnist -- --retrain        # entrena desde cero
```

`--predict` muestra las diez salidas, para que puedas ver qué otras clases recibieron
probabilidades significativas:

```
  This is a 4. The network says 4 — correct.

    4  0.999  ███████████████████████████████████████
    9  0.001
```

Después de guardar, la demostración recarga el archivo y comprueba que 1000 predicciones salen
**idénticas bit a bit**. Un serializador que pierde o reordena parámetros puede producir un modelo
que se carga y produce predicciones, pero cuyo rendimiento se ha degradado: la misma clase de fallo
silencioso que un gradiente incorrecto.

El `dataset` no está en este repositorio. La demostración lo descarga una vez (~11 MB) y lo guarda
en caché fuera del árbol de trabajo; toda ejecución posterior, incluidas las que no tienen conexión,
lee esa caché. Si no hay red ni caché, muestra un mensaje explicativo y finaliza de forma
controlada. Así `dotnet test` y la otra demostración nunca dependen de que un servidor réplica del
`dataset` esté disponible.

### Leer un dígito desde un archivo de imagen

El repositorio sí incluye un reconocedor ya entrenado:
[`models/mnist-784-128-10.nnm`](models/). Por eso `--image` funciona en un clon recién hecho, sin
entrenar nada y sin descargar el `dataset`:

```
Image:  my-digit.png
  248x248 image, normalized to MNIST's 28x28 convention:

                        ..****++
                    ==@@@@@@@@@@++
                  ..@@@@%%--..@@@@..
                  ++@@%%      ::@@++

  This is a 0.  (confidence 0.999)
```

PNG y Netpbm se decodifican en [`ImageFile.cs`](src/NN.Mnist/ImageFile.cs) sin dependencias externas
al framework de .NET. La mayor parte de ese archivo no es descompresión —de eso se encarga
`ZLibStream`—, sino la reversión de los filtros por fila que usa PNG.

**El decodificador es la parte fácil.** La parte importante es
[`DigitPreprocessor`](src/NN.Mnist/DigitPreprocessor.cs), porque la red no aprendió «dígitos» en
abstracto: aprendió las convenciones de MNIST. Eso significa tinta blanca sobre fondo negro,
escalado dentro de una caja de 20×20 y centrado en 28×28 *por centro de masa*. Si la imagen no
respeta alguna de esas convenciones, la precisión se desploma de una forma que parece un modelo
roto. Esas cien líneas valen tanto como los 101 770 parámetros entrenados, y la guía de estudio
explica en detalle por qué.

Esto se verificó de extremo a extremo exportando dígitos de prueba de MNIST como PNG de 248×248,
oscuro sobre claro y con márgenes amplios, incumpliendo así las tres convenciones. Al leerlos de
vuelta, **10 de 10 coincidieron con lo que el modelo predice sobre los datos crudos**, incluido uno
en el que se *equivoca*. Reproducir fielmente los errores del modelo demuestra que el
preprocesamiento es transparente y no está ayudando por accidente.

### Softmax y cross-entropy: impacto en el entrenamiento

La demostración clasifica con una **salida `softmax` y pérdida `cross-entropy`**: la configuración
estándar cuando debes elegir una sola clase entre varias mutuamente excluyentes. Esa es una de las
razones por las que llega al 98 %. La versión anterior, con diez salidas sigmoidales independientes
evaluadas mediante MSE, sigue disponible para comparar:

```bash
dotnet run -c Release --project src/NN.Mnist -- --loss mse --retrain
```

| | Precisión en test | Learning rate necesario |
|---|---|---|
| MSE sobre diez salidas sigmoidales | 97,41 % | **1,0** |
| Softmax + cross-entropy | **98,02 %** | **0,1** |

Misma arquitectura, mismas 20 épocas, misma semilla. Con el mismo `learning rate` de 0,1, la
versión con MSE solo alcanza el 92,93 %. Necesita 1,0 para compensar un gradiente que la propia
función de pérdida ya redujo demasiado.

**Por qué.** MSE sobre sigmoides trata los dígitos como diez preguntas de sí/no sin relación entre
sí, y su gradiente arrastra un factor `σ'(z) = a(1−a)` que se desploma hacia cero justo cuando la
red produce una predicción incorrecta con alta confianza —precisamente cuando más necesita
aprender—. `Softmax` hace que las diez salidas *compitan* (suman 1), y `cross-entropy` solo puntúa
la probabilidad asignada a la clase correcta.

Derivadas por separado, `softmax` produce un Jacobiano completo y `cross-entropy` introduce un
término `1/p`; **compuestas, casi todo se cancela y el gradiente queda simplemente como `p − y`**:
la probabilidad predicha menos la etiqueta objetivo, sin factores que se desvanezcan ni cálculos
que se desborden.

Esa cancelación solo es válida sobre `logits` crudos, así que `SoftmaxOutput()` construye una capa
lineal `Dense<Identity>`. La pérdida rechaza una capa de salida ya comprimida, en lugar de calcular
un gradiente silenciosamente incorrecto. El gradiente fusionado se verifica contra diferencias
finitas en la batería de pruebas: un atajo algebraico casi correcto es exactamente el tipo de error
para el que existe [`GradientCheck`](src/NN/GradientCheck.cs).

El margen que queda es el SGD simple, sin momento ni Adam (guía de estudio §25 punto 2, ejercicio
10), con **98,02 % en 37 s** como la marca a batir.

## Qué está implementado

| | |
|---|---|
| **Capas** | Densa (totalmente conectada), de cualquier profundidad |
| **Activaciones** | Sigmoid, Tanh, ReLU, Identity, Step |
| **Entrenamiento** | Backpropagation, SGD por mini-batches, barajado |
| **Pérdidas** | MSE; softmax + cross-entropy para clasificación |
| **Inicialización** | Xavier/Glorot uniforme |
| **API del modelo** | Constructor `Sequential` estilo Keras, `Summary()` |
| **Persistencia** | Formato binario versionado con arquitectura + pesos; entrena una vez, recarga en ms |
| **Verificación** | `Gradient check` por diferencias finitas |
| **Datos** | Dataset «dos lunas» generado; cargador de MNIST (formato IDX, descarga + caché) |
| **Imágenes** | Decodificación de PNG y Netpbm, y normalización a las convenciones de MNIST, sin dependencias |

## Notas de diseño

Cada afirmación de esta sección está medida; los números están en [`bench/`](bench/README.es.md).

**Los pesos se almacenan por unidad en un único arreglo plano.** Los pesos de la unidad `j` ocupan
`Weights[j * Inputs .. (j+1) * Inputs]`, es decir, la transpuesta de la disposición `(n, j)` de
NumPy. Eso convierte cada producto escalar en un recorrido SIMD contiguo en lugar de un acceso no
contiguo, y vuelve a ayudar en `backpropagation`, donde tanto la acumulación del gradiente de los
pesos como el gradiente que vuelve hacia la entrada recorren la misma memoria contigua.
**Medido: 6,2–7,4× en capas de 64×64 y 784×128**, el mayor efecto del código. En la capa XOR de 2×4
no ayuda, porque sus ocho pesos caben en una sola línea de caché; ahí la versión no contigua es
ligeramente más rápida.

**SIMD se adapta a la CPU.** [`SimdOps`](src/NN/SimdOps.cs) usa `Vector<float>`, que tiene ancho 4
en ARM NEON y 8 en AVX2. Usa dos acumuladores para mantener ocupado el `pipeline` de
multiplicación-suma y una cola escalar para longitudes que no son múltiplo del ancho. **Medido:
4,1–5,9× frente a un bucle escalar, y el segundo acumulador aporta otro 1,2–1,5×.**

**Uno de los dos primitivos SIMD es del runtime; el otro no.** `AddScaled` llama a
`TensorPrimitives.MultiplyAdd` de `System.Numerics.Tensors` y es **2,5× más rápido** por ello.
`Dot` conserva su propio bucle porque `TensorPrimitives.Dot` midió **1,5× más lento**: arrastra una
sola cadena de acumulación a través de la reducción, que es la dependencia serie que el segundo
acumulador existe para romper. Misma biblioteca, respuestas opuestas, decididas
[midiendo y no por reputación](bench/README.es.md#por-qué-no-llamar-directamente-a-tensorprimitivesdot).

**La activación se aplica a toda una capa a la vez, no unidad por unidad.** `exp` y `tanh` cuestan
decenas de ciclos cada uno; vectorizados se calculan de cuatro en cuatro. **Medido: 2× en el paso de
activación, 1,3× en la capa.** Conseguirlo requirió un
`[MethodImpl(MethodImplOptions.NoInlining)]`, sin el cual el mismo código es **6× más lento**: el
JIT deja de eliminar las comprobaciones de límites en cuanto el bucle se inserta en línea dentro de
un método genérico grande. Esa historia
[merece leerse](bench/README.es.md#la-regresión-de-6-que-provocó-este-cambio-y-su-solución-de-una-palabra): una
optimización local hizo seis veces más lento justo aquello que optimizaba, y solo el benchmark lo
detectó.

**Las activaciones se representan mediante parámetros de tipo genérico, no mediante delegados.**
`Dense<Tanh>` usa miembros de interfaz estáticos abstractos de C# 11, así que el JIT puede realizar
la inserción en línea (*inlining*) de la función de activación dentro del bucle. Este README antes
afirmaba que la alternativa obvia —un campo `Func<float, float>`— sería claramente más lenta por
hacer una llamada indirecta por unidad. **La medición dice que no: queda dentro de ±4 % en tamaños
realistas, sin un signo consistente.**

La activación se ejecuta una vez por *unidad*, mientras que el producto escalar que la alimenta
ejecuta `Inputs` multiplicaciones-sumas, así que el coste de la llamada queda amortizado. El diseño
genérico se mantiene por sus méritos reales —composición sin coste con activaciones
`readonly struct` y sin asignar delegados—, pero no por velocidad.

**El `forward pass` de inferencia y el de entrenamiento son métodos separados.** `Forward` calcula
activaciones; `ForwardTrain` además guarda en caché lo que `backpropagation` necesita. Cuando un
solo método hacía ambas cosas, cualquier `forward pass` incidental —evaluar una pérdida, registrar
una predicción a mitad de época— sobrescribía la caché en silencio. El siguiente `Backward`
calculaba entonces gradientes para el ejemplo equivocado sin dar error. Separar los métodos hace que
ese estado inválido no se pueda representar, y `Backward` lanza una excepción si no lo precedió un
`ForwardTrain`.

## Hilos y propiedad de buffers

La biblioteca **está diseñada para ejecutarse en un solo hilo y reutiliza buffers internos**. De ahí
salen dos reglas:

- **Una red por hilo.** `Network` y `Dense` mantienen buffers de activación mutables, acumuladores
  de gradiente y el estado del barajado. Nada está sincronizado. (`Dense.Forward` sobre una capa
  independiente no toca estado de instancia; `Network.Predict` sí escribe buffers de activación
  compartidos.)
- **`Predict` devuelve una vista de un buffer que la siguiente llamada sobrescribe.** Si necesitas
  conservar el resultado, o varias predicciones simultáneamente, copia cada uno con `.ToArray()`.

`ModelIO.Register` modifica una tabla global del proceso. El acceso a esa tabla está sincronizado,
pero sigue siendo más claro registrar tipos de capa personalizados durante el arranque, no mientras
otro código está cargando modelos.

## Estructura del proyecto

```
src/NN/           la biblioteca
src/NN.Demo/      demostración ejecutable — perceptrón, XOR, dos lunas
src/NN.Mnist/     reconocimiento de dígitos manuscritos, con lector IDX y caché del dataset
tests/NN.Tests/   la batería de pruebas
bench/NN.Bench/   benchmarks; resultados en bench/README.md (inglés) y bench/README.es.md
STUDY-GUIDE.md    la explicación extensa (inglés)
STUDY-GUIDE.es.md la explicación extensa (español)
```

## Requisitos

SDK de .NET 10. [`global.json`](global.json) fija la versión mayor, de modo que una máquina con
varios SDK instalados compile este repositorio con aquel contra el que fue probado.

La biblioteca depende de un único paquete, `System.Numerics.Tensors`, que aporta los `kernels`
vectorizados de `AddScaled` descritos en las notas de diseño. El proyecto de pruebas usa xUnit y los
benchmarks usan BenchmarkDotNet.

## Licencia

MIT — véase [LICENSE](LICENSE).
