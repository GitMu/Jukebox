#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <dirent.h>
#include <Fmodex/fmod.h>



using namespace cv;
using namespace std;


/** @function main */
int main()
{
    int L=20,S=10,H=2;

    unsigned char isFile =0x8; //code to test if it is a file (not a folder)
    string dossier="/home/pi/Documents/Programmation/Jukebox/Medias/"; // location of pictures
    string fichimage;

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    //-- Step 2: Calculate descriptors (feature vectors)
    OrbDescriptorExtractor extractor;

    std::vector<KeyPoint> keypoints_scene;

    std::vector<std::vector<KeyPoint>> tableau_keypoint; // vector of keypoints
    std::vector<Mat> tableau_image; // vector of the pictures
    std::vector<Mat> tableau_descripteur;
    SurfFeatureDetector detector( minHessian ); //create the detector. Surf give good results compared to Orb
    std::vector<string> tableau_fichier; //put the filenames in a vector

    // FMOD Initialization to play sound
    FMOD_SYSTEM *system;
    FMOD_SOUND *son;
    FMOD_RESULT resultat;
    FMOD_CHANNEL *channel;
    FMOD_System_Create(&system);
    FMOD_System_Init(system, 1, FMOD_INIT_NORMAL, NULL);
    FMOD_System_GetChannel(system,9,&channel);
    string justemp3=  dossier + "Audio/juste.wav" ; //audio file for good result
    string fauxmp3=  dossier + "Audio/faux.wav" ;//audio file for bad result


    int trouve,livre=0;
    Mat capture;
    char key;

    cout << "Taper 'q' pour quitter" << endl;

    //handle webcam
    VideoCapture cap(0); // open the video camera no. 0

    if (!cap.isOpened())  // if not success, exit program
        {
            cout << "Cannot open the video cam" << endl;
            return -1;
        }


    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);//create a window called "MyVideo"

    createTrackbar( "L", "MyVideo", &L, 40 );
    createTrackbar( "S", "MyVideo", &S, 30 );
    createTrackbar( "H", "MyVideo", &H, 20 );


    // Open pictures and create keypoints for each to put in a vector
    DIR * rep = opendir(dossier.c_str());

        if (rep != NULL)
            {
            struct dirent * ent;
            while ((ent = readdir(rep)) != NULL)
                {
                if ( ent->d_type == isFile) //we check if it is a file or a folder
                    {
                    fichimage=dossier+ent->d_name;
                    Mat img_object = imread( fichimage, CV_LOAD_IMAGE_GRAYSCALE ); //read the picture in gray
                    Mat descriptors_object;

                    if( !img_object.data)
                        {
                        cout<< " --(!) Error reading images " << std::endl;
                        printf("%s\n", ent->d_name);
                        return -1;// To be changed: if there is an issue with a file the program will bug
                        }

                    std::vector<KeyPoint> keypoints_object;
                    tableau_fichier.push_back(ent->d_name);
                    tableau_image.push_back(img_object);
                    detector.detect( img_object, keypoints_object );
                    tableau_keypoint.push_back(keypoints_object);// On envoie les point dans le tableau
                    extractor.compute( img_object, keypoints_object, descriptors_object);
                    tableau_descripteur.push_back(descriptors_object);
                    }
                }
            closedir(rep);
            }


    while (1)
    {

        Mat floue,frame;

        bool bSuccess = cap.read(frame ); // read a new frame from video

         if (!bSuccess) //if not success, break loop
            {
             cout << "Cannot read a frame from video stream" << endl;
             break;
            }
        if (key == 32) // take a capture when we press 'space'
            {
            Mat img_scene, edged, closed;

            //some works on picture to increase accuracy and speed: transform in gray and add blur
            cvtColor(frame, floue, COLOR_BGR2GRAY);
            GaussianBlur(floue, img_scene, cv::Size(3,3), 0);

            //detection of keypoints of the screen capture then comparison with database
            detector.detect( img_scene, keypoints_scene );
            Mat descriptors_scene;
            extractor.compute( img_scene, keypoints_scene, descriptors_scene );
            FlannBasedMatcher matcher(new flann::LshIndexParams(L, S, H)); //Flann matcher (20,10,2)
            std::vector< DMatch > matches;

            double meilleur=100; //variable to stock the best candidate's distance
            for (int it=0;it<tableau_descripteur.size();it++)
                {
                matcher.match( tableau_descripteur[it], descriptors_scene, matches );
                double max_dist = 0; double min_dist = 100; // min_dist's value to be adapted. Low enough to avoid false positive but high enough to found something

                //-- Quick calculation of max and min distances between keypoints
                for( int i = 0; i < tableau_descripteur[i].rows; i++ )
                    { double dist = matches[i].distance;
                    if( dist < min_dist ) min_dist = dist;
                    if( dist > max_dist ) max_dist = dist;
                    }

                    // if the min distance (ie best matching) is lower than the best (meilleur) distance then we have a better candidate
                    if (meilleur > min_dist)
                        {
                        trouve = it; //we keep index of the best candidate
                        meilleur=min_dist;
                        livre=1; //to be modified when there is no matching
                        }
                    std::cout << tableau_fichier[it] << "  " << min_dist <<"  " << max_dist << endl;
                    }

            if (meilleur<30)        //if we find something. to be modified when there is no matching
                {
                //we play a little sound to say we found it

                resultat = FMOD_System_CreateSound(system, justemp3.c_str(), FMOD_CREATESAMPLE, 0, &son);
                FMOD_System_PlaySound(system,son,0,0,&channel);
                std::cout << "Trouvé " << tableau_fichier[trouve] << "  " << meilleur << endl;

                // We show the result found in a new window
                Mat cover;
                string imgresult= dossier + "Archive/" +tableau_fichier[trouve] ;
                cover = imread(imgresult, CV_LOAD_IMAGE_COLOR);
                namedWindow("Trouvé",CV_WINDOW_AUTOSIZE);
                imshow("Trouvé", cover);



                //and we play the audio file which has the same name (but with mp3 extension) in a folder called /Audio

                string nombase= tableau_fichier[trouve];
                nombase.resize(nombase.size()-3); //remove .jpg extension
                string fichiermp3= dossier + "Audio/" + nombase + "mp3";

                resultat = FMOD_System_CreateStream(system, fichiermp3.c_str(), FMOD_CREATESAMPLE, 0, &son);//as it is a mp3 we use CreateStream instead of CreateSound: much faster to start

                if (resultat != FMOD_OK)

                    {
                    cout << "Erreur. Impossible de lire : " << fichiermp3 << endl;
                    }

                FMOD_System_PlaySound(system,son,0,0,&channel);

                }
            else //if we found nothing we play a wrong audio
                {
                cout << meilleur << endl;
                resultat = FMOD_System_CreateSound(system, fauxmp3.c_str(), FMOD_CREATESAMPLE, 0, &son);
                FMOD_System_PlaySound(system,son,0,0,&channel);
                }

            }
        imshow("MyVideo", frame); //show the frame in "MyVideo" window



    key = (char) cv::waitKey(30);   // explicit cast
    if (key == 113) break;                // break if `q' key was pressed.

    }
    //release of the stuff regarding Audio
    FMOD_Sound_Release(son);
    FMOD_System_Close(system);
    FMOD_System_Release(system);
    return 0;

}

