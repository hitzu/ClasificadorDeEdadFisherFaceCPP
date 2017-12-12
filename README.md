# Estimador de edad con Fisherfaces

Preguntas y comentarios: [robert.torres.lopez@gmail.com]

¿Como installar?
--------------
1. Instalar [Visual Studio 2013]
2. Instalar Qt VS2013 
http://download.qt.io/official_releases/qt/5.6/5.6.1/qt-opensource-windows-x86-msvc2013-5.6.1.exe
3. Installar openCV 
http://opencv.org/downloads.html
4. Agregar C:\opencv\build\x86\vc12\bin a las variables de entorno
5. Probar que todo este bien.

Este tutorial puede verse en: http://spanishopencv.blogspot.mx/2016/06/instalando-opencv-2413-qt-creator-en.html


Notes
--------------
Asegurate de instalar la version 2013 de visual studio, si no tendras bastantes problemas para corregir ese paso

Descargar el programa
--------------
1. Clona este repositorio.
2. Crea los archivos menores.txt y mayores.txt
3. Dentro de los archivos colocaras el nombre y la extension de los archivos con los cuales entrenaras al clasificador.
4. Las imagenes correspondientes a cada archivo deberan colocarse DENTRO de la misma carpeta donde se encuentra el archivo.
5. En el metodo LeerImagenes(vector<Mat>& imagenes, string seleccion) se debera cambiar la ruta a donde apunta el String path, hacia la ruta local donde se encuentran la carpeta con los archivos e imagens (Ejemplo en estructura de carpetas).
6. Correr el programa y verificar que no exista ningun problema.


## Estructura de carpetas
- Clasificador
  - imagenes
    - mayores.txt: Aqui se encuentran los nombres y extensiones de las imagenes para entrenar el grupo de los mayores de edad
    - menores.txt: Aqui se encuentran los nombres y extensiones de las imagenes para entrenar el grupo de los menores de edad
    - imagen1Mayores.jpg
    - imagen2Mayores.jpg
    - imagen1Menores.jpg
    - imagen2Menores.jpg

## Base de imagenes
1. Fg-net https://drive.google.com/open?id=0B8K1LEKpDxZoRW1YUndLb0pkazA
2. IMBD Wiki en grupos de 20 elementos por edad en un rango de 0 a 100 https://drive.google.com/open?id=0B8K1LEKpDxZoM0Y1X2RyZWltSVU
