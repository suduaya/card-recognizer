#include "opencv2/opencv.hpp"
#include <time.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "opencv2/core/core.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

///global templates
Mat naipes[3];
Mat card_symbols[13];
void MatchingMethod(Mat card);

double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


void loadtempls() {
	///naipes
	/*
		0 -> ouros,
		1 -> copas,
		2 -> espadas,
		3 -> paus,
		4 -> Ás ouros
	*/
	naipes[0] = imread("naipes/Diamonds.png");
	naipes[1] = imread("naipes/Hearts.png");
	naipes[2] = imread("naipes/Spades.png");
	naipes[3] = imread("naipes/Clubs.png");
	naipes[4] = imread("naipes/As.png");
	///simbolos
	/*
		0 -> Ás,
		1 -> 2,
		2 -> 3,
		3 -> 4,
		4 -> 5,
		5 -> 6,
		6 -> 7,
		7 -> 8,
		8 -> 9,
		9 -> 10,
		10 -> Q,
		11 -> J,
		12 -> K
	*/
	card_symbols[0] = imread("cards/As.png");
	card_symbols[1] = imread("cards/dois.png");
	card_symbols[2] = imread("cards/tres.png");
	card_symbols[3] = imread("cards/quatro.png");
	card_symbols[4] = imread("cards/cinco.png");
	card_symbols[5] = imread("cards/seis.png");
	card_symbols[6] = imread("cards/sete.png");
	card_symbols[7] = imread("cards/oito.png");
	card_symbols[8] = imread("cards/nove.png");
	card_symbols[9] = imread("cards/dez.png");
	card_symbols[10] = imread("cards/dama.png");
	card_symbols[11] = imread("cards/valete.png");
	card_symbols[12] = imread("cards/rei.png");
	return;
}

int main(int, char**)
{
	loadtempls();
	VideoCapture cap(1); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	for (;;) {
		Mat edges;
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		cv::threshold(edges, edges, 128, 255, CV_THRESH_BINARY);
		std::vector<std::vector<cv::Point> > contours;
		std::vector<Mat> cards = vector<Mat>();;
		cv::Mat contourOutput = edges.clone();
		cv::findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		cv::Mat contourImage(edges.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		///new
		std::vector<std::vector<cv::Point2f> > squares;

		std::vector<cv::Point2f> approx;
		for (size_t idx = 0; idx < contours.size(); idx++) {
			cv::approxPolyDP(cv::Mat(contours[idx]), approx, cv::arcLength(cv::Mat(contours[idx]), true)*0.02, true);
			if (approx.size() == 4 && std::fabs(contourArea(cv::Mat(approx))) > 1000 &&
				cv::isContourConvex(cv::Mat(approx)))
			{
				double maxCosine = 0;
				for (int j = 2; j < 5; j++)
				{
					double cosine = std::fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
					maxCosine = MAX(maxCosine, cosine);
				}

				if (maxCosine < 0.3)
					squares.push_back(approx);
			}
			
		}
		
		if (squares.size() > 0) {
			
			for(size_t ii = 0 ; ii< squares.size(); ii++){
				RotatedRect rekt;
				int bx = 0, by = 0, sx = 150000, sy = 150000;
				int ibx = 0, iby = 0, isx = 0 , isy = 0;
				for (size_t i = 0; i < squares[ii].size(); i++) {
					cv::circle(frame, squares[ii][i], 4, cv::Scalar(0, 0, 255), cv::FILLED);
					if (squares[ii][i].x < sx) {
						sx = squares[ii][i].x;
						isx = i;
					}
					if (squares[ii][i].y < sy) {
						sy = squares[ii][i].y;
						isy = i;
					}
					if (squares[ii][i].x > bx) {
						bx = squares[ii][i].x;
						ibx = i;
					}
					if (squares[ii][i].y > by) {
						by = squares[ii][i].y;
						iby = i;
					}
				}
				///
				/// è preciso criar o rotated rect
				rekt = minAreaRect(Mat(squares[ii]));
				///
				Mat M, rotated, cropped;
				// get angle and size from the bounding box
				float angle = rekt.angle;
				Size rect_size = rekt.size;
				if (rekt.angle < -45.) {
					angle += 90.0;
					swap(rect_size.width, rect_size.height);
				}
				// get the rotation matrix
				M = getRotationMatrix2D(rekt.center, angle, 1.0);
				// perform the affine transformation
				warpAffine(frame, rotated, M, frame.size(), INTER_CUBIC);
				// crop the resulting image
				getRectSubPix(rotated, rect_size, rekt.center, cropped);
				try {
					cards.push_back(cropped);
				}
				catch (...) {}

				
			}
			for (int i = 0; i < cards.size(); i++) {
				Mat to_gray;
				/// gray and blur
				cvtColor(cards[i], to_gray, COLOR_BGR2GRAY);
				blur(to_gray, to_gray, Size(3, 3));
				//erosao
				Mat imageEroded;
				Mat structElem = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
				erode(to_gray, imageEroded, structElem, Point(1, 1), 5);

				Mat canny_output;
				vector<Vec4i> hierarchy;
				RNG rng(12345);
				/// Detect edges using canny
				int value = 40;  // we manualy tested out
				Canny(to_gray, canny_output, value, value * 2, 3);

				/// Find contours
				findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

				/// Draw contours
				Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
				for (size_t i = 0; i< contours.size(); i++)
				{
					Scalar color = cv::Scalar(255, 255, 255);
					drawContours(drawing, contours, (int)i, color, 1, 8, hierarchy, 0, Point());
				}
				MatchingMethod(drawing);
			}
			
			
				
		}
		//imshow("canny", contourImage);
		imshow("frame", frame);
		if (waitKey(30) >= 0) break;
	
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
		
}


string naipe_res[4] = {
	"Ouros",
	"Copas",
	"Espadas",
	"Paus"
};

string cards_res[13] = {
	"As",
	"Dois",
	"Tres",
	"Quatro",
	"Cinco",
	"Seis",
	"Sete",
	"Oito",
	"Nove",
	"Dez",
	"Dama",
	"Jota",
	"Rei"
};

void MatchingMethod(Mat card)
{
	imshow("card", card);
	int naipe = -1;
	for (int i = 0; i < 5; i++) {
		try
		{
			/// Load image and template
			Mat img;
			Mat templ;
			naipes[i].copyTo(templ);
			

			cvtColor(card, card, CV_BGR2HSV);
			cvtColor(templ, templ, CV_BGR2HSV);
			threshold(card, card, 255, 255, 255);
			threshold(templ, templ, 255, 255, 255);
			//imshow("card", card);
			

			/// Source image to display
			Mat result;
			/// Create the result matrix
			int result_cols = card.cols - templ.cols + 1;
			int result_rows = card.rows - templ.rows + 1;

			result.create(result_rows, result_cols, CV_32FC1);

			/// Do the Matching and Normalize
			int match_method = CV_TM_SQDIFF_NORMED;
			matchTemplate(card, templ, result, match_method);
			normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

			/// Localizing the best match with minMaxLoc
			double minVal; double maxVal; Point minLoc; Point maxLoc;
			Point matchLoc;

			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

			/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
			if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
			{
				matchLoc = minLoc;
			}
			else
			{
				matchLoc = maxLoc;
			}

			/// Show me what you got
			rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(70), 2, 8, 0);

			

			if (matchLoc.x != 0) {
				if (i == 4) {
					cout << "encontrei As de ouros" << endl;
					return;
				}
				cout << "encontrei naipe" <<  naipe_res[naipe] << endl;
				naipe = i;
			}
		}
		catch (const std::exception&)
		{
			cout << "crashou kek" << endl;
		}
		
	}
	if (naipe == -1)
		return;
	///simbolos
	for (int i = 0; i < 13; i++) {
		try
		{
			/// Load image and template
			Mat img;
			Mat templ;
			card_symbols[i].copyTo(templ);


			cvtColor(card, card, CV_BGR2HSV);
			cvtColor(templ, templ, CV_BGR2HSV);
			threshold(card, card, 255, 255, 255);
			threshold(templ, templ, 255, 255, 255);
			//imshow("card", card);


			/// Source image to display
			Mat result;
			/// Create the result matrix
			int result_cols = card.cols - templ.cols + 1;
			int result_rows = card.rows - templ.rows + 1;

			result.create(result_rows, result_cols, CV_32FC1);

			/// Do the Matching and Normalize
			int match_method = CV_TM_SQDIFF_NORMED;
			matchTemplate(card, templ, result, match_method);
			normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

			/// Localizing the best match with minMaxLoc
			double minVal; double maxVal; Point minLoc; Point maxLoc;
			Point matchLoc;

			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

			/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
			if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
			{
				matchLoc = minLoc;
			}
			else
			{
				matchLoc = maxLoc;
			}

			/// Show me what you got
			rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(70), 2, 8, 0);



			if (matchLoc.x != 0) {
				cout << "Encontrei " << cards_res[i] << " de " << naipe_res[naipe] << endl;
			}
		}
		catch (const std::exception&)
		{
			cout << "crash" << endl;
		}

	}


}