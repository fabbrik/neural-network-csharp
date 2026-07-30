# Red neuronal desde cero en C#

*[English](README.md) · **Español***

Una red neuronal de alimentación hacia adelante (*feed-forward*) construida desde cero en C# — sin
bibliotecas de machine learning, sin frameworks, solo arreglos (`arrays`) y cálculo.
Backpropagation, matemáticas aceleradas con SIMD, serialización de modelos, verificación de
gradientes y benchmarks para cada afirmación de velocidad.

Viene acompañada de una [**guía de estudio**](STUDY-GUIDE.es.md) que explica *por qué existe cada
pieza*: desde «qué es una neurona», pasando por la regla de la cadena, hasta las líneas de caché.
También incluye dos ejemplos resueltos que puedes seguir con una calculadora.

> **Convención de términos.** Cuando el término técnico se usa normalmente en inglés entre
> programadores o estudiantes de IA (`forward pass`, `mini-batch`, `learning rate`, `benchmark`,
> `dataset`), esta versión lo deja en inglés y lo explica en español cuando ayuda. Evita
> traducciones literales que suenan correctas pero que casi nadie usa en código, papers o cursos.

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

## ¿Por qué otro repositorio de estos?

Hay muchos repositorios de «red neuronal desde cero». Tres cosas aquí son menos comunes:

**Verifica su propia backpropagation.** Una implementación sutilmente incorrecta de
backpropagation puede seguir entrenando *hasta cierto punto*, lo que hace que estos errores sean
difíciles de encontrar.
[`GradientCheck`](src/NN/GradientCheck.cs) compara cada gradiente analítico contra una diferencia
finita central, y la batería de pruebas afirma la curva en U de precisión que solo produce un
gradiente *correcto* — incluyendo una prueba con una derivada rota a propósito, para demostrar que
la verificación puede efectivamente fallar.

**Mide sus propias afirmaciones, y reporta la que resultó errónea.** Cada afirmación de rendimiento
en esta documentación tiene un [benchmark](bench/) detrás. Una de ellas — que una
activación llamada mediante un delegado sería significativamente más lenta que una genérica —
resultó ser **falsa**, y
[el informe lo dice](bench/README.md#3-generic-activation-vs-delegate--the-claim-that-was-wrong)
en lugar de descartarla en silencio.

**Es honesta sobre sus límites.** [El §25 de la guía de estudio](STUDY-GUIDE.es.md) enumera lo que
esta implementación no hace y lo que costaría, en orden — empezando por admitir que un GEMM por
lotes superaría cualquier optimización SIMD del código, una afirmación que los benchmarks ahora
[confirman parcialmente](bench/README.md#4-forwardbatch--a-deliberate-null-result).

## Cómo ejecutarlo

```bash
dotnet run --project src/NN.Demo      # perceptrón, XOR, guardar/cargar, gradiente, dos lunas
dotnet run -c Release --project src/NN.Mnist   # reconocimiento de dígitos manuscritos (~40 s)
dotnet test                           # la batería completa de pruebas
dotnet run -c Release --project bench/NN.Bench -- --filter '*'   # benchmarks
```

> **Sobre el idioma de la salida.** Los programas imprimen en inglés, así que los bloques de salida
> que aparecen abajo se reproducen tal cual: son lo que verás realmente en tu terminal. Traducirlos
> aquí sería describir un programa que no existe. El código, los nombres de identificadores y las
> banderas de línea de comandos se mantienen igual por la misma razón.

La demo ejecuta seis secciones. Las primeras cuatro son a escala XOR — el perceptrón convergiendo
en AND, una capa oculta resolviendo XOR, un modelo que va y vuelve del disco, y una verificación de
gradiente:

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

Las dos últimas usan un dataset lo bastante grande como para mostrar lo que XOR no puede mostrar
por su estructura: mini-batches, generalización a datos no vistos y sobreajuste:

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
> ancho 4 en ARM pero 8 en AVX2 — así que el producto escalar SIMD suma en distinto orden en
> distintas CPU. Los números citados aquí y en la guía de estudio provienen de un Apple M3 Pro con
> .NET 10. Espera que los últimos dígitos cambien en otro hardware; espera que las conclusiones no.
> La integración continua se ejecuta deliberadamente en ambas arquitecturas.

## Leer dígitos manuscritos

Una demo aparte entrena con [MNIST](src/NN.Mnist/) — 60 000 dígitos manuscritos, 784 entradas,
101 770 parámetros. **98,0 % sobre dígitos que nunca ha visto, en 37 segundos**, con dos capas
densas, backpropagation y una salida softmax:

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

Después imprime los dígitos en los que se **equivocó**. Eso explica mejor el «98 %» que el número
por sí solo: la mayoría son lo bastante ambiguos como para que una persona también dudara.

**El modelo entrenado se guarda, así que solo pagas el entrenamiento una vez.** Volver a
ejecutarlo carga 397 KB de pesos en lugar de reentrenar: **37 s → 4 ms**, con el mismo 98,02 %.

```bash
dotnet run -c Release --project src/NN.Mnist                     # reutiliza el modelo guardado
dotnet run -c Release --project src/NN.Mnist -- --predict 42     # clasifica una imagen de prueba
dotnet run -c Release --project src/NN.Mnist -- --image d.png    # clasifica tu propia imagen
dotnet run -c Release --project src/NN.Mnist -- --retrain        # entrena desde cero
```

`--predict` muestra las diez salidas, para que veas lo que estuvo a punto de decir:

```
  This is a 4. The network says 4 — correct.

    4  0.999  ███████████████████████████████████████
    9  0.001
```

Tras guardar, la demo recarga el archivo y comprueba que 1000 predicciones salen **idénticas bit a
bit**. Un serializador que pierde o reordena parámetros puede producir un modelo que todavía carga
y todavía predice, pero que simplemente es *peor*: la misma clase de fallo silencioso que un
gradiente incorrecto.

El dataset no está en este repositorio. La demo lo descarga una vez (~11 MB) y lo guarda
en caché fuera del árbol de trabajo; toda ejecución posterior, incluidas las que no tienen
conexión, lee esa caché. Sin red y sin caché lo explica y termina limpiamente en lugar de fallar —
de modo que `dotnet test` y la otra demo nunca dependen de que un servidor espejo del conjunto de
datos esté disponible.

### Leer un dígito desde un archivo de imagen

Sí hay un reconocedor entrenado incluido en el repositorio —
[`models/mnist-784-128-10.nnm`](models/) — de modo que `--image` funciona en un clon recién hecho,
sin entrenar nada y sin descargar el dataset:

```
Image:  my-digit.png
  248x248 image, normalized to MNIST's 28x28 convention:

                        ..****++
                    ==@@@@@@@@@@++
                  ..@@@@%%--..@@@@..
                  ++@@%%      ::@@++

  This is a 0.  (confidence 0.999)
```

PNG y Netpbm se decodifican en [`ImageFile.cs`](src/NN.Mnist/ImageFile.cs) sin más dependencias que
el framework — y la mayor parte de ese archivo no es descompresión (de eso se encarga `ZLibStream`)
sino la reversión de los filtros por fila que usa PNG.

**El decodificador es la parte fácil.** La parte importante es
[`DigitPreprocessor`](src/NN.Mnist/DigitPreprocessor.cs), porque la red no aprendió «dígitos» —
aprendió las convenciones de MNIST: tinta blanca sobre negro, escalado dentro de un recuadro de
20×20, centrado en 28×28 *por centro de masa*. Viola cualquiera de ellas y la precisión se desploma
de una forma que parece exactamente un modelo roto. Esas cien líneas valen tanto como los 101 770
parámetros entrenados, y la guía de estudio explica en detalle por qué.

Se verificó de extremo a extremo exportando dígitos de prueba de MNIST como PNG de 248×248, oscuro
sobre claro y con márgenes amplios — rompiendo las tres convenciones — y leyéndolos de vuelta:
**10 de 10 coincidieron con lo que el modelo predice sobre los datos crudos**, incluido uno en el
que se *equivoca*. Reproducir fielmente los errores del modelo demuestra que el preprocesado es
transparente y no está ayudando por accidente.

### Softmax y entropía cruzada, y cuánto valen

La demo clasifica con una **salida softmax y pérdida de entropía cruzada** — la configuración
estándar para elegir entre categorías mutuamente excluyentes, y la razón por la que alcanza el
98 %. La versión anterior, diez sigmoides independientes puntuadas con MSE, sigue disponible para
comparar:

```bash
dotnet run -c Release --project src/NN.Mnist -- --loss mse --retrain
```

| | Precisión en prueba | Learning rate necesaria |
|---|---|---|
| MSE sobre diez sigmoides | 97,41 % | **1,0** |
| Softmax + entropía cruzada | **98,02 %** | **0,1** |

Misma arquitectura, mismas 20 épocas, misma semilla. Con un `learning rate` *idéntico* de 0,1
la versión con MSE solo alcanza el 92,93 %: necesita 1,0 solo para compensar un gradiente que la
propia función de pérdida ya redujo demasiado.

**Por qué.** MSE sobre sigmoides trata los dígitos como diez preguntas de sí/no sin relación entre
sí, y su gradiente arrastra un factor `σ'(z) = a(1−a)` que se desploma hacia cero justo cuando la
red está equivocada con confianza — precisamente cuando más necesita aprender. Softmax hace que las
diez salidas *compitan* (suman 1), y la entropía cruzada puntúa solo la probabilidad asignada a la
respuesta correcta. Derivadas por separado, softmax da un jacobiano completo y la entropía cruzada
un `1/p` que explota; **compuestas, casi todo se cancela y el gradiente es simplemente `p − y`** —
predicción menos objetivo, sin factores que se desvanezcan ni cálculos que se desborden.

Esa cancelación solo es válida sobre logits crudos, así que `SoftmaxOutput()` construye una capa
lineal `Dense<Identity>` y la pérdida *rechaza* una capa comprimida en lugar de calcular un
gradiente silenciosamente incorrecto. El gradiente fusionado se verifica contra diferencias finitas
en la batería de pruebas: un atajo algebraico que es casi correcto es exactamente el tipo de error
para el que existe [`GradientCheck`](src/NN/GradientCheck.cs).

El margen que queda es el SGD simple, sin momento ni Adam (guía de estudio §25 punto 2, ejercicio
10), con **98,02 % en 37 s** como la marca a batir.

## Qué está implementado

| | |
|---|---|
| **Capas** | Densa (totalmente conectada), de cualquier profundidad |
| **Activaciones** | Sigmoid, Tanh, ReLU, Identity, Step |
| **Entrenamiento** | Backpropagation, SGD por mini-batches, shuffling |
| **Pérdidas** | Error cuadrático medio; softmax + entropía cruzada para clasificación |
| **Inicialización** | Xavier/Glorot uniforme |
| **API del modelo** | Constructor `Sequential` estilo Keras, `Summary()` |
| **Persistencia** | Formato binario versionado con arquitectura + pesos; entrena una vez, recarga en ms |
| **Verificación** | Comprobación de gradiente por diferencias finitas |
| **Datos** | Dataset «dos lunas» generado; cargador de MNIST (formato IDX, descarga + caché) |
| **Imágenes** | Decodificación de PNG y Netpbm, y normalización a las convenciones de MNIST, sin dependencias |

## Notas de diseño

Cada una de estas fue medida; los números enlazan a [`bench/`](bench/README.md).

**Los pesos se almacenan por unidad en un único arreglo plano.** Los pesos de la unidad `j`
ocupan `Weights[j * Inputs .. (j+1) * Inputs]` — la transpuesta de la disposición `(n, j)` de
NumPy. Eso convierte cada producto escalar en un recorrido SIMD contiguo en lugar de una lectura
dispersa, y vuelve a ayudar en backpropagation, donde tanto la acumulación del gradiente de
los pesos como la propagación del gradiente de entrada recorren la misma memoria contigua.
**Medido: 4,6–5,9× en capas de 64×64 y 784×128** — el mayor efecto del código. No vale *nada* en la
capa XOR de 2×4, cuyos ocho pesos caben en una sola línea de caché; ahí, la versión dispersa es
marginalmente más rápida.

**SIMD se adapta a la CPU.** [`SimdOps`](src/NN/SimdOps.cs) usa `Vector<float>`, que tiene ancho 4
en ARM NEON y 8 en AVX2, con dos acumuladores para mantener alimentada la tubería de
multiplicación-suma y una cola escalar para longitudes que no son múltiplo del ancho. **Medido:
4,7–6,2× frente a un bucle escalar, y el segundo acumulador aporta otro 1,2–1,5×** en cualquier
longitud mayor que un par de vectores — por debajo de eso cuesta un poco, cosa que el informe de
rendimiento no oculta.

**Las activaciones son parámetros de tipo genéricos, no delegados.** `Dense<Tanh>` usa miembros de
interfaz estáticos abstractos de C# 11, así que el JIT integra la activación dentro del bucle. Este
README solía afirmar que la alternativa obvia — un campo `Func<float, float>` — costaría una
llamada indirecta que el JIT no puede integrar por unidad y por tanto sería más lenta. **No se
puede integrar, pero no es más lenta: se midió dentro de ±2 % en cualquier tamaño realista, y ni siquiera con signo
consistente.** La activación se ejecuta una vez por *unidad* mientras que el producto escalar que
la alimenta ejecuta `Inputs` multiplicaciones-sumas, así que la llamada queda amortizada hasta la
invisibilidad. El diseño genérico se mantiene, por sus méritos reales — composición a coste cero
con activaciones `readonly struct` y sin asignación de delegados — pero no por velocidad.

**El `forward pass` de inferencia y el de entrenamiento son métodos separados.**
`Forward` calcula activaciones; `ForwardTrain` además guarda en caché lo que backpropagation
necesita. Cuando un solo método hacía ambas cosas, cualquier forward pass incidental
— evaluar una pérdida, registrar una predicción a mitad de época — sobrescribía la caché en
silencio, de modo que el siguiente `Backward` derivaba el ejemplo equivocado sin dar error. La
separación hace que ese estado inválido no se pueda representar, y `Backward` lanza una excepción
si no lo precedió un `ForwardTrain`.

## Hilos y propiedad de los buffers

La biblioteca es **de un solo hilo por diseño y presta sus buffers**. Dos reglas:

- **Una red por hilo.** `Network` y `Dense` mantienen buffers de activación mutables, acumuladores
  de gradiente y el estado del barajado. Nada está sincronizado. (`Dense.Forward` sobre una capa
  independiente no toca estado de instancia; `Network.Predict` sí escribe buffers de activación
  compartidos.)
- **`Predict` devuelve una vista que la siguiente llamada sobrescribe.** Cópiala con `.ToArray()`
  si necesitas conservarla, y nunca retengas dos resultados de predicción a la vez.

`ModelIO.Register` modifica una tabla global del proceso. El acceso a la tabla está sincronizado,
pero registrar tipos de capa personalizados durante el arranque sigue siendo más claro que dejar
que el comportamiento de carga dependa del momento en que ocurra.

## Estructura del proyecto

```
src/NN/           la biblioteca
src/NN.Demo/      demostración ejecutable — perceptrón, XOR, dos lunas
src/NN.Mnist/     reconocimiento de dígitos manuscritos, con lector IDX y caché del dataset
tests/NN.Tests/   la batería de pruebas
bench/NN.Bench/   benchmarks, con resultados en bench/README.md
STUDY-GUIDE.md    la explicación extensa (inglés)
STUDY-GUIDE.es.md la explicación extensa (español)
```

## Requisitos

SDK de .NET 10. [`global.json`](global.json) fija la versión mayor, de modo que una máquina con
varios SDK instalados compile este repositorio con aquel contra el que fue probado.

La biblioteca no tiene dependencias más allá del framework; el proyecto de pruebas usa xUnit y los
benchmarks usan BenchmarkDotNet.

## Licencia

MIT — véase [LICENSE](LICENSE).
