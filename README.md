# digit-recognition
This repository contains a simple GUI that allows the user to draw simple pictures. If the user presses a key after drawing their picture, a Residual Neural Network will attempt to identify the digit that was entered.

The intent of this repository is as an educational device for teaching the basics of Deep Learning Development to non-experts. The model can be trained by running the model.py script. This will write the metrics produced during training to the lightning_logs folder. These metrics can be visualised using tensorboard if so desired.

By running the interface.py script, a window will appear that allows the user to draw on the screen. Once the user is finished with their drawing, they can press any key and the model will attempt to identify the digits in the image. Its predictions will be visualised with a bar chart.

Users will discover that simply entering well drawn digits will usually result in good predictions by the model. However, entering poorly drawn digits or rotated digits will result in poor predictions. This serves as a good demonstration of the uses and dangers of neural networks when applied in the real world: They can be very powerful but they are also only as good as the data upon which they are trained.
