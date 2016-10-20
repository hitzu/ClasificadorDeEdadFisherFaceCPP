//bibliotecas opencv
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

//bibliotecas c++
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

int num_components;
cv::Mat eigenvectors;
cv::Mat eigenvalues;
cv::Mat mean1;
cv::Mat labels;
std::vector<cv::Mat> projections;
std::vector<int> etiquetas;

vector<Mat> LeerImagenes(vector<Mat>& imagenes, string seleccion)
{
    cout << "Iniciando carga de imagenes: " << seleccion << endl;
    char cadena[128];
    //Aqui ponemos el path al archivo con las fotos de los menores de edad hasta 18 años
    string path = "C:\\Users\\Hitzu\\Documents\\proyectosQT\\clasificador\\train\\";
    //leyendo el archivo
    ifstream fe("C:\\Users\\Hitzu\\Documents\\proyectosQT\\clasificador\\train\\"+seleccion);
    while(!fe.eof())
    {
        fe.getline(cadena,128);
        path = path + cadena;
        imagenes.push_back(imread(path,CV_LOAD_IMAGE_COLOR));
        //cout << path << endl;
        path = "C:\\Users\\Hitzu\\Documents\\proyectosQT\\clasificador\\train\\";
    }
    fe.close();
    return imagenes;
}

vector<Mat> LeerImagenesTest(string seleccion)
{
    vector<Mat> imagenes;
    cout << "Iniciando carga de imagenes de prueba: " << seleccion << endl;
    char cadena[128];
    //Aqui ponemos el path al archivo con las fotos de los menores de edad hasta 18 años
    string path = "C:\\Users\\Hitzu\\Documents\\proyectosQT\\clasificador\\test\\";
    //leyendo el archivo
    ifstream fe("C:\\Users\\Hitzu\\Documents\\proyectosQT\\clasificador\\test\\"+seleccion);
    while(!fe.eof())
    {
        fe.getline(cadena,128);
        path = path + cadena;
        imagenes.push_back(imread(path,CV_LOAD_IMAGE_COLOR));
        //cout << path << endl;
        path = "C:\\Users\\Hitzu\\Documents\\proyectosQT\\clasificador\\test\\";
    }
    fe.close();
    return imagenes;
}


//Filtros diferentes

//canny
Mat canny(Mat imagen)
{
    Mat gray, edge, draw;
    cvtColor(imagen, gray, CV_BGR2GRAY);

    Canny( gray, edge, 50, 150, 3);

    edge.convertTo(draw, CV_8U);

    return draw;
}

Mat laplace(Mat imagen)
{
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    GaussianBlur(imagen, imagen,Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(imagen, imagen, CV_BGR2GRAY);
    Laplacian(imagen,imagen, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( imagen, imagen );

    return imagen;
}

Mat sobel (Mat imagen)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat grad;
    GaussianBlur(imagen, imagen,Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(imagen, imagen, CV_BGR2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(imagen, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    Sobel(imagen, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    return grad;
}

Mat transformadaH(Mat imagen)
{
    Mat dst, cdst;
    Canny(imagen, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
    Vec4i l = lines[i];
    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
    return cdst;
}

vector<Mat> preprocesamiento(vector<Mat>& imagenes)
{
    cout << "Iniciando procesamiento de las imagenes " << endl;
    vector<Mat> procesadas;
    Mat modificada, recortada;
    int eliminadas;
    Vector<int> posiciones;
    for(int i = 0; i < imagenes.size(); i ++)
    {
        cvtColor(imagenes[i], modificada, CV_BGR2GRAY);
        equalizeHist(modificada,modificada);
        modificada = imagenes[i];
        CascadeClassifier face_cascade;
        face_cascade.load( "C://opencv//sources//data//haarcascades//haarcascade_frontalface_alt2.xml" );

        std::vector<Rect> faces;
        face_cascade.detectMultiScale( modificada, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        for( int j = 0; j < faces.size(); j++ )
        {
            //recortamos la parte de interes (cara)
            Rect myROI( faces[j].x, faces[j].y, (faces[j].width), (faces[j].height) );
            recortada = modificada(myROI);
            cv::resize(recortada, recortada, cv::Size(211,211));
        }
        if(faces.size() == 1)
        {
            //los filtros se colocaran en esta parte
            //recortada = canny(recortada);
            //recortada = laplace(recortada);
            //recortada = sobel(recortada);
            //recortada = transformadaH(recortada);
            procesadas.push_back(recortada);
        }
        else
        {
            posiciones.push_back(i);
            //etiquetas.erase(etiquetas.begin()+i);
            //eliminadas++;
            //cout << eliminadas << endl;
        }
    }
    //eliminando etiquetas
    for( int j = 0; j < posiciones.size(); j++ )
    {
        etiquetas.erase(etiquetas.begin()+(posiciones[j]-j));
        //cout << "eliminadas" << endl;
        //cout << posiciones[j] << endl;
    }

    return procesadas;
}

Mat asRowMatrix(const vector<Mat>& src, int rtype) {
    double alpha = 1;
    double beta = 0;
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    cout << "tamaño de la matriz " << n << endl;
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    cout << "tamaño de las muestras: " << d << endl;
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            //string error_message = format("La imagen numero %d esta vacia por favor revisar los datos", i);
            //CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Numero equivocados de elementos en la matriz #%d! se esperaban %d y fueron %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

Mat subspaceProject(Mat W, Mat mean, Mat src) {
    // get number of samples and dimension
    int n = src.rows;
    int d = src.cols;
    // make sure the data has the correct shape
    if(W.rows != d) {
        string error_message = format("Wrong shapes for given matrices. Was size(src) = (%d,%d), size(W) = (%d,%d).", src.rows, src.cols, W.rows, W.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    // make sure mean is correct if not empty
    if(!mean.empty() && (mean.total() != d)) {
        string error_message = format("Wrong mean shape for the given data matrix. Expected %d, but was %d.", d, mean.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    // create temporary matrices
    Mat X, Y;
    // make sure you operate on correct type
    src.convertTo(X, W.type());
    // safe to do, because of above assertion
    // safe to do, because of above assertion
    if(!mean.empty()) {
        for(int i=0; i<n; i++) {
            Mat r_i = X.row(i);
            subtract(r_i, mean.reshape(1,1), r_i);
        }
    }
    // finally calculate projection as Y = (X-mean)*W
    gemm(X, W, 1.0, Mat(), 0.0, Y);
    return Y;
}

void train(vector<Mat>& src, InputArray _lbls) {

    if(src.size() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(CV_StsBadArg, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(CV_StsBadArg, error_message);
    }
    // make sure data has correct size
    if(src.size() > 1) {
        for(int i = 1; i < static_cast<int>(src.size()); i++) {
            if(src[i-1].total() != src[i].total()) {
                string error_message = format("In the method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", src[i-1].total(), src[i].total());
                CV_Error(CV_StsUnsupportedFormat, error_message);
            }
        }
    }
    // get data
    Mat labelss = _lbls.getMat();
    Mat data = asRowMatrix(src, CV_64FC1);
    // number of samples
    int N = data.rows;
    // make sure labels are passed in correct shape
    if(labelss.total() != (size_t) N) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", N, labelss.total());
        CV_Error(CV_StsBadArg, error_message);
    } else if(labelss.rows != 1 && labelss.cols != 1) {
        string error_message = format("Expected the labels in a matrix with one row or column! Given dimensions are rows=%s, cols=%d.", labelss.rows, labelss.cols);
       CV_Error(CV_StsBadArg, error_message);
    }
    // clear existing model data
    labels.release();
    projections.clear();
    // safely copy from cv::Mat to std::vector
    vector<int> ll;
    for(unsigned int i = 0; i < labelss.total(); i++) {
        ll.push_back(labelss.at<int>(i));
    }
    // get the number of unique classes
    //int C = (int) remove_dups(ll).size();
    // clip number of components to be a valid number
    //if((num_components <= 0) || (num_components > (C-1)))
    //    num_components = (C-1);
    // perform a PCA and keep (N-C) components
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, /*(N-C)*/num_components);
    // project the data and perform a LDA on it
    LDA lda(pca.project(data),labelss, num_components);
    // store the total mean vector
    mean1 = pca.mean.reshape(1,1);
    // store labels
    labels = labelss.clone();
    // store the eigenvalues of the discriminants
    lda.eigenvalues().convertTo(eigenvalues, CV_64FC1);
    // Now calculate the projection matrix as pca.eigenvectors * lda.eigenvectors.
    // Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
    gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, eigenvectors, GEMM_1_T);
    // store the projections of the original data
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = subspaceProject(eigenvectors, mean1, data.row(sampleIdx));
        projections.push_back(p);
    }
}

int predict(Mat _src) {
    Mat src = _src;
    // check data alignment just for clearer exception messages
    if(projections.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This model is not computed yet. Did you call train?";
        CV_Error(CV_StsBadArg, error_message);
    } else if(src.total() != (size_t) eigenvectors.rows) {
        string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", eigenvectors.rows, src.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    // project into LDA subspace
    Mat q = subspaceProject(eigenvectors, mean1, src.reshape(1,1));
    // find 1-nearest neighbor
    double minDist = DBL_MAX;
    double threshold = DBL_MAX;
    int minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < projections.size(); sampleIdx++) {
        double dist = norm(projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < threshold)) {
            minDist = dist;
            minClass = labels.at<int>((int)sampleIdx);
        }
    }
    return minClass;
}


int main()
{
    int i;
    vector<Mat> imagenes;
    imagenes = LeerImagenes(imagenes,"menores.txt");
    int tammenores = imagenes.size();
    for(i = 0; i < tammenores; i++)
    {
        etiquetas.push_back(0);
    }
    imagenes = LeerImagenes(imagenes,"mayores.txt");
    int tammayores = imagenes.size();
    for(i = tammenores; i < tammayores; i++)
    {
        etiquetas.push_back(1);
    }

    imagenes = preprocesamiento(imagenes);
    train(imagenes,etiquetas);
    cout << "se acaba la creacion de modelos " << endl;
    //empieza fase de prediccion
    vector<Mat> menores;
    menores = LeerImagenesTest("menores.txt");
    vector<Mat> mayores;
    mayores = LeerImagenesTest("mayores.txt");

    menores = preprocesamiento(menores);
    mayores = preprocesamiento(mayores);

    //recorriendo los arreglos de imagenes para predicciones
    int resultado;
    int correcto = 0;
    int incorrecto = 0;
    int correctos_menores = 0;
    int correctos_mayores = 0;

    for(i = 0; i < menores.size(); i++)
    {
        resultado = predict(menores[i]);
        if(resultado == 0)
        {
            correctos_menores++;
            correcto++;
        }
        else
        {
            incorrecto++;
        }
    }

    //limpiando el resultado
    resultado = -1;
    for(i = 0; i < mayores.size(); i++)
    {
        resultado = predict(mayores[i]);
        if(resultado == 1)
        {
            correctos_mayores++;
            correcto++;
        }
        else
        {
            incorrecto++;
        }
    }

    cout << "Con " << imagenes.size() << " imagenes de entrenamiento" << endl;
    cout << "de tamaño " << menores[0].rows << " pixeles se obtiene" << endl;

    cout << "Hay: " << correctos_menores << " aciertos y " << menores.size() - correctos_menores  << " errores en el grupo de los menores de: " << menores.size() << endl;
    cout << "Hay: " << correctos_mayores << " aciertos y " << mayores.size() - correctos_mayores << " errores en el grupo de los mayores de: " << mayores.size() << endl;

    //los valores


    int total = menores.size() + mayores.size();
    cout << "La muestra total es de: " << total << " con: " << correcto << " aciertos y " << incorrecto <<  "errores" << endl;

    double porcentaje = (correcto*100)/total;
    cout << "Lo que nos da una efectividad del: " << porcentaje << " Por ciento en PCA + LDA" << endl;


    waitKey(0);
    return 0;
}
