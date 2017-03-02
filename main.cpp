#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/opencv.hpp"
#include <sys/socket.h> 
#include <arpa/inet.h>
#include <unistd.h>
#include <raspicam/raspicam_cv.h>
int height =240;
int width =320;
int i=1;
using namespace cv;
using namespace std;

void detectFace(Mat img,Mat img_final);
void onMouse( int event, int x, int y, int, void* );
void BMSetMinDisp(int val, void* ptr)
{	
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	if(bm->state->minDisparity >= 0)
		bm->state->minDisparity = val;
	else
		bm->state->minDisparity = -val;
}

void BMflipMinDisp(int val, void* ptr)
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->minDisparity = -bm->state->minDisparity;
}

void BMSetNumDisp( int val, void* ptr )
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->numberOfDisparities = (val + 1) *16  ;
}

void BMSetPreFilterCap( int val , void* ptr )
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->preFilterCap = val + 1;
}

void BMSetSAD( int val, void* ptr )
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	if(val < 5)
		val = 5;
	if(val > 255)
		val = 255;

	if(val % 2 != 1)
		bm->state->SADWindowSize = val + 1;
	else
		bm->state->SADWindowSize = val;
	
}
void BMSetTextureThresh( int val, void* ptr )
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->textureThreshold = val;
}
void BMSetUniqueness( int val, void* ptr )

{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->uniquenessRatio = val;
}

void BMSetSpeckleWindowSize(int val, void* ptr)
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->speckleWindowSize = val;
}

void BMSetSpeckleRange(int val, void* ptr)
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->speckleRange = val;
}

void BMSetMaxDiff(int val, void* ptr)
{
	StereoBM *bm= static_cast<StereoBM*>(ptr);
	bm->state->disp12MaxDiff = static_cast<float>(val/100.);
}
void SGBMSetPreFilterCap(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	bm->preFilterCap = val;
}

void SGBMSetSADWindowSize(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	if(val % 2 == 1)
	{
		bm->SADWindowSize = val;
		bm->P1 = 8*3*val*val;
		bm->P2 = 32*3*val*val;
	}
	else
	{
		val = val+1;
		bm->SADWindowSize = val;
		bm->P1 = 8*3*val*val;
		bm->P2 = 32*3*val*val;
	}
}

void SGBMSetMinDisparity(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	bm->minDisparity = -val;
}

void SGBMSetNumberOfDisparities(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	bm->numberOfDisparities = val*16;
}

void SGBMSetUniquenessRatio(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	bm->uniquenessRatio = val;
}

void SGBMSetSpeckleWindowSize(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	bm->speckleWindowSize = val;
}

void SGBMSetSpeckleRange(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	bm->speckleRange = val*16;
}
		
void SGBMSetMaxDiff(int val, void* ptr)
{
	StereoSGBM *bm = static_cast<StereoSGBM*>(ptr);
	bm->disp12MaxDiff = val;
}

void VarSetMinDisp(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	bm->minDisp = -val;
}

void VarSetMaxDisp(int val, void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	bm->maxDisp = -val;
}

void VarSetLevels(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	bm->levels = val;
}

void VarSetPyrScale(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	if(val != 0)
		bm->pyrScale = val/10.0f;
	
}


void VarSetNIts(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	bm->nIt = val;
}

void VarSetPoly_n(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	if(val <= 3)
		bm->poly_n = 3;
	else if (val > 3 && val <= 5)
		bm->poly_n = 5;
	else
		bm->poly_n = 7;
}

void VarSetPoly_sigma(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	bm->minDisp = val/100.0f;
}

void VarSetFi(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	bm->fi = val/10.0f;
}

void VarSetLambda(int val ,void* ptr)
{
	StereoVar *bm = static_cast<StereoVar*>(ptr);
	bm->lambda = val/100.0f;;
}


Mat disp, disp8;
Mat before_median;
Mat & after_median = before_median;

string face_cascade_name = "haarcascade_frontalface_alt.xml";  
CascadeClassifier face_cascade;   

int main(int argc, char** argv)
{

    //--------------------------------------------------------
    //networking stuff: socket , connect
    //--------------------------------------------------------
    int         sokt;
    char*       serverIP;
    int         serverPort;

    if (argc < 3) {
           std::cerr << "Usage: cv_video_cli <serverIP> <serverPort> " << std::endl;
    }

    serverIP   = argv[1];
    serverPort = atoi(argv[2]);

    struct  sockaddr_in serverAddr;
    socklen_t           addrLen = sizeof(struct sockaddr_in);

    if ((sokt = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "socket() failed" << std::endl;
    }

    serverAddr.sin_family = PF_INET;
    serverAddr.sin_addr.s_addr = inet_addr("192.168.0.11");
    serverAddr.sin_port = htons(4097);

    if (connect(sokt, (sockaddr*)&serverAddr, addrLen) < 0) {
        std::cerr << "connect() failed!" << std::endl;
    }



    //----------------------------------------------------------
    //OpenCV Code
    //----------------------------------------------------------
	raspicam::RaspiCam_Cv Camera;
	//set camera params
	Camera.set( CV_CAP_PROP_FORMAT, CV_8UC1 );
	Camera.set(CV_CAP_PROP_FRAME_WIDTH, width);
	Camera.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	//Open camera
	cout<<"Opening Camera..."<<endl;

    	if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;}
	namedWindow("trackbars", CV_WINDOW_AUTOSIZE);
	namedWindow("Kamera na pokladzie",1);
	namedWindow("Kamera zdalna",1);
	namedWindow("Disp",1);
	
	moveWindow("Kamera na pokladzie", 0, 0);
	moveWindow("Kamera zdalna", 480, 0);
	moveWindow("Disp", 0, 360);
	moveWindow("trackbars", 480, 360);

	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;
	
	int BMTrackMinDisp = 39;
	int BMTrackFlipMinDisp = 1;
	int BMTrackNumberOfDisparities = 7;
	int BMTrackPreFilterCap = 61;
	int BMTrackSADWindowSize = 9;
	int BMTrackTextureThreshold = 507;
	int BMTrackUniquenessRatio = 0;
	int BMTrackSpeckleWindowSize = 0;
	int BMTrackSpeckleRange = 8;
	int BMTrackDisp12MaxDiff = 100;
	
	int SGBMPreFilterCap = 0;
	int SGBMSADWindowSize = 0;
	int SGBMP1 = 8*3*SGBMSADWindowSize*SGBMSADWindowSize;
	int SGBMP2 = 32*3*SGBMSADWindowSize*SGBMSADWindowSize;
	int SGBMMinDisparity = 0;
	int SGBMNumberOfDisparities = 5;
	int SGBMUniquenessRatio = 0;
	int SGBMSpeckleWindowSize = 150;
	int SGBMSpeckleRange = 2;
	int SGBMMaxDiff = 10;

	int VarLevels = 1;
	int VarPyrScale = 17;
	int VarnIt = 12;
	int VarPoly_n = 3;
	int VarPoly_sigma = 0;
	int VarFi = 80;
	int VarLambda = 20;
	int VarNumberOfDisparities = 0;

	Mat img, img2;
	Mat right, left, g1, g2, imgU1, imgU2;
	Mat img1_mat;
	Mat lmapx, lmapy, rmapx, rmapy;
	Mat R1, R2, P1, P2, Q;
	Mat K1, K2, R;
	Vec3d T;
	Mat D1, D2;
	cv::FileStorage fs1("cam_stereo.yml", cv::FileStorage::READ);
	fs1["K1"] >> K1;
	fs1["K2"] >> K2;
	fs1["D1"] >> D1;
	fs1["D2"] >> D2;
	fs1["R"] >> R;
	fs1["T"] >> T;
	
	fs1["R1"] >> R1;
	fs1["R2"] >> R2;
	fs1["P1"] >> P1;
	fs1["P2"] >> P2;
	fs1["Q"] >> Q;
	stringstream jeden , dwa;


	
	if(i==1)
	{
		bm.state->preFilterCap = BMTrackPreFilterCap;
		bm.state->SADWindowSize = BMTrackSADWindowSize;
		bm.state->minDisparity = BMTrackMinDisp;
		bm.state->numberOfDisparities = BMTrackNumberOfDisparities*16;
		bm.state->textureThreshold = BMTrackTextureThreshold;
		bm.state->uniquenessRatio = BMTrackUniquenessRatio;
		bm.state->speckleWindowSize = BMTrackSpeckleWindowSize;
		bm.state->speckleRange = BMTrackSpeckleRange;
		bm.state->disp12MaxDiff = BMTrackDisp12MaxDiff/100.;
	
		createTrackbar("minDisp", "trackbars", &BMTrackMinDisp, 16, BMSetMinDisp, &bm);
		createTrackbar("flipminDisp","trackbars", &BMTrackFlipMinDisp, 1, BMflipMinDisp, &bm);
		createTrackbar("numDisp","trackbars", &BMTrackNumberOfDisparities , 20 , BMSetNumDisp, &bm);
		createTrackbar("preFilterCap","trackbars", &BMTrackPreFilterCap , 62, BMSetPreFilterCap, &bm);
		createTrackbar("SADWindowSize","trackbars", &BMTrackSADWindowSize , 255, BMSetSAD, &bm);
		createTrackbar("TextureThresh","trackbars", &BMTrackTextureThreshold , 1000, BMSetTextureThresh, &bm);
		createTrackbar("UniquenessRatio","trackbars", &BMTrackUniquenessRatio , 100, BMSetUniqueness, &bm);
		createTrackbar("SpeckleWindowSize","trackbars", &BMTrackSpeckleWindowSize , 200, BMSetSpeckleWindowSize, &bm);
		createTrackbar("SpeckleRange","trackbars", &BMTrackSpeckleRange , 100, BMSetSpeckleRange, &bm);
		createTrackbar("MaxDiff","trackbars", &BMTrackDisp12MaxDiff , 2000, BMSetMaxDiff, &bm);
	}
	else if(i==2)
	{
		sgbm.preFilterCap = SGBMPreFilterCap;
		sgbm.SADWindowSize = SGBMSADWindowSize;
		sgbm.P1 = SGBMP1;
		sgbm.P2 = SGBMP2;
		sgbm.minDisparity = SGBMMinDisparity;
		sgbm.numberOfDisparities = SGBMNumberOfDisparities*16;
		sgbm.uniquenessRatio = SGBMUniquenessRatio;
		sgbm.speckleWindowSize = SGBMSpeckleWindowSize;
		sgbm.speckleRange = SGBMSpeckleRange*16;
		sgbm.disp12MaxDiff = SGBMMaxDiff;
		sgbm.fullDP = false;
	
		createTrackbar("PreFilter","trackbars", &SGBMPreFilterCap , 100, SGBMSetPreFilterCap , &sgbm);
		createTrackbar("SADWindowSize","trackbars", &SGBMSADWindowSize , 30, SGBMSetSADWindowSize , &sgbm);
		createTrackbar("MinDisparity","trackbars", &SGBMMinDisparity , 16, SGBMSetMinDisparity , &sgbm);
		createTrackbar("NumberOfDisparities","trackbars", &SGBMNumberOfDisparities , 16, SGBMSetNumberOfDisparities , &sgbm);
		createTrackbar("UniquenessRatio","trackbars", &SGBMUniquenessRatio , 30, SGBMSetUniquenessRatio , &sgbm);
		createTrackbar("SpeckleWindowSize","trackbars", &SGBMSpeckleWindowSize , 300, SGBMSetSpeckleWindowSize , &sgbm);
		createTrackbar("SpeckleRange","trackbars", &SGBMSpeckleRange , 20, SGBMSetSpeckleRange , &sgbm);
		createTrackbar("MaxDiff","trackbars", &SGBMMaxDiff , 100, SGBMSetMaxDiff , &sgbm);	
	}
	else if(i==3)
	{
		//Initialization if alg = StereoVar
		var.levels = VarLevels;
		var.pyrScale = VarPyrScale/10.0f;
		var.nIt = VarnIt;
		var.minDisp = -VarNumberOfDisparities;
		var.maxDisp = 0;
		var.poly_n = VarPoly_n;
		var.poly_sigma = VarPoly_sigma/100.;;
		var.fi = VarFi/10.;
		var.lambda = VarLambda/100.;
		var.penalization = var.PENALIZATION_TICHONOV;
		var.cycle = var.CYCLE_V;
		var.flags = cv::StereoVar::USE_SMART_ID | cv::StereoVar::USE_AUTO_PARAMS | cv::StereoVar::USE_INITIAL_DISPARITY | cv::StereoVar::USE_MEDIAN_FILTERING;
	
		createTrackbar("minDisp", "trackbars", &VarNumberOfDisparities,32 , VarSetMinDisp, &var);
		createTrackbar("levels", "trackbars", &VarLevels, 4, VarSetLevels, &var );	
		createTrackbar("pyrScale", "trackbars", &VarPyrScale, 100, VarSetPyrScale, &var );	
		createTrackbar("nIts", "trackbars", &VarnIt, 50, VarSetNIts, &var );	
		createTrackbar("poly_n", "trackbars", &VarPoly_n, 7, VarSetPoly_n, &var );	
		createTrackbar("poly_sigma", "trackbars", &VarPoly_sigma, 2200, VarSetPoly_sigma, &var );	
		createTrackbar("fi", "trackbars", &VarFi, 200, VarSetFi, &var );	
		createTrackbar("lambda", "trackbars", &VarLambda, 100, VarSetLambda, &var );	
	}
	string s;
	stringstream out; 
	int x,y;  
	int event;
	int odl;	
	ostringstream ss;
	img = Mat::zeros(height , width, CV_8UC1);    
	img2 = Mat::zeros(height , width, CV_8UC1);
	int imgSize = img.total() * img.elemSize();
	uchar *iptr = img.data;
	int bytes = 0;
	int key;
	
	std::cout << "Image Size:" << imgSize << std::endl;
	
	cv::initUndistortRectifyMap(K1, D1, R1, P1, img.size(), CV_32F, lmapx, lmapy);
	cv::initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32F, rmapx, rmapy);

    setMouseCallback( "Disp", onMouse, 0 );
    while (key != 'q') {
        if ((bytes = recv(sokt, iptr, imgSize , MSG_WAITALL)) == -1) {
            std::cerr << "recv failed, received bytes = " << bytes << std::endl;
        }
	Camera.grab();
	Camera.retrieve (img2);	
	clock_t start = clock();

	//cv::remap(img, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
	//cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
	
	if (i==1)
	{
	bm(img2, img, disp);
	}
	else if(i==2)
	{
	sgbm(img, img2, disp);
	}
	else if(i==3)
	{
	var(img, img2, disp);
	}
	normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
	medianBlur(disp8, after_median, 9);
	detectFace(img, after_median);
	printf("Czas wczytywania: %.2fs\n ",(double)(clock() -start)/CLOCKS_PER_SEC);
	cv::imshow("Disp", after_median);
        cv::imshow("Kamera na pokladzie", img2);
        cv::imshow("Kamera zdalna", img); 
	if(key=cv::waitKey(10)>=0)break;
    }   

    close(sokt);

    return 0;
}

void detectFace(Mat img,Mat img_final)
{
	if (!face_cascade.load(face_cascade_name))        //£adowanie pliku ze sprawdzeniem poprawnoci 
	{
		cout << "Nie znaleziono pliku " << face_cascade_name << ".";
	}
	vector<Rect> faces;

	face_cascade.detectMultiScale(img, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50));
	for (unsigned i = 0; i < faces.size(); i++)
	{
		Rect rect_face(faces[i]);    //Kwadrat okreslaj¹cy twarz 
									 //ellipse( img, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 120, 12 ), 2, 2, 0 ); 
		rectangle(img_final, rect_face, Scalar(120, 5, 86), 2, 2, 0);
	}
	         
}


void onMouse( int event, int x, int y, int, void* )
{
    	if( event != CV_EVENT_LBUTTONDOWN )
    	return;
    	Point pt = Point(x,y);
    	cout<<"x="<<pt.x<<"\t y="<<pt.y<<"\t value="<<int( after_median.at<uchar>(y,x) )<<"\n";
    	int odl=0.0758*(after_median.at<uchar>(y,x)*after_median.at<uchar>(y,x))-28.592*(after_median.at<uchar>(y,x))+2750.1;
    	cout<<"odleglosc= "<<odl<<"\n"; 

}
