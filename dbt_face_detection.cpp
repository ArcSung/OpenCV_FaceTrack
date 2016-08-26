#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID)

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <stdio.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

const string WindowName = "Face Detection example";

int PeopleCount = 0;
int lastPeopleCount = 0;
int tempPeopleCount = 0;

class _People{

public:
    int Faceindex;
    int Peopleindex;
    Point locat;

};

class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
    public:
        CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
            IDetector(),
            Detector(detector)
        {
            CV_Assert(detector);
        }

        void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
        {
            Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
        }

        virtual ~CascadeDetectorAdapter()
        {}

    private:
        CascadeDetectorAdapter();
        cv::Ptr<cv::CascadeClassifier> Detector;
 };

int main(int , char** )
{
    namedWindow(WindowName);

    VideoCapture VideoStream(0);

    if (!VideoStream.isOpened())
    {
        printf("Error: Cannot open video stream from camera\n");
        return 1;
    }

    std::string cascadeFrontalfilename = "haarcascade_frontalface_alt.xml";
    cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);

    cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);

    DetectionBasedTracker::Parameters params;
    DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

    if (!Detector.run())
    {
        printf("Error: Detector initialization failed\n");
        return 2;
    }

    Mat ReferenceFrame;
    Mat GrayFrame;
    vector<Rect> Faces;
    vector<_People> People;

    VideoStream >> ReferenceFrame;
    Mat TrackMat = Mat(ReferenceFrame.size(), CV_8UC1, Scalar(0));

    while(true)
    {
        VideoStream >> ReferenceFrame;
        cvtColor(ReferenceFrame, GrayFrame, COLOR_RGB2GRAY);
        Detector.process(GrayFrame);
        Detector.getObjects(Faces);

        if(People.size() != Faces.size())
        {
            if(People.size() > Faces.size())
            {
                printf("People.size() > Faces.size()\n");
                int s = 0;
                while(People.size() > Faces.size())
                {
                    if(TrackMat.at<unsigned char>(People[s].locat.y, People[s].locat.y) == 0)
                        People.erase(People.begin() + s);
                    else
                        ++s;
                }    
            }
            else if(People.size() < Faces.size()) //add People
            {
                printf("People.size() < Faces.size()\n");
                int diff = Faces.size() - People.size();
                int s = Faces.size();
                for(int i = 0; i < diff; i++)
                {
                    ++PeopleCount;
                    _People tempPeople;
                    int tempPindex =  PeopleCount;
                    int tempFindex =  i;
                    tempPeople.Peopleindex = tempPindex;
                    tempPeople.Faceindex = tempFindex;
                    tempPeople.locat = Point(Faces[i+s].x + Faces[i+s].width*0.5, Faces[i+s].y + Faces[i+s].height*0.5);
                    People.push_back(tempPeople);

                }    

            }    
        }    

        TrackMat.setTo(Scalar(0));
        for (size_t i = 0; i < Faces.size(); i++)
        {
            rectangle(ReferenceFrame, Faces[i], Scalar(0,255,0), 3);
            rectangle(TrackMat, Faces[i], Scalar(People[i].Peopleindex), CV_FILLED, 8, 0);
            string box_text = format("Tracked Area %d", People[i].Peopleindex);
            int pos_x = std::max(Faces[i].tl().x - 10, 0);
            int pos_y = std::max(Faces[i].tl().y - 10, 0);
            // And now put it into the image:
            putText(ReferenceFrame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0,255,0), 2.0);
            
            //update
            People[i].locat = Point(Faces[i].x + Faces[i].width*0.5, Faces[i].y + Faces[i].height*0.5);
        }

        //imshow("TrackMat", TrackMat);
        imshow(WindowName, ReferenceFrame);

        if (waitKey(30) >= 0) break;
    }

    Detector.stop();

    return 0;
}

#else

#include <stdio.h>
int main()
{
    printf("This sample works for UNIX or ANDROID only\n");
    return 0;
}

#endif
