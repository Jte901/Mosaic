#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

// Find identifying features between two images using the SURF algorithm
//
// Pre: imgs contains non empty Mat images
//
// @arg imgs:         vector of Mat images
// @arg num_images:   number of images to stitch
// @arg work_megapix: choose default to be 0.6
// @arg seam_megapix: choose default to be 0.1
// @arg seam_work_aspect: choose default to be 1
//
// Returns: Vector of structures containing image keypoints and descriptors.
//
vector<ImageFeatures> findFeatures(vector<Mat> imgs, int num_images,
									double work_megapix, double seam_megapix,
									double &seam_work_aspect) {
	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false,
			is_compose_scale_set = false;

	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);

	for (int i = 0; i < num_images; ++i) {
		full_img = imgs[i];
		full_img_sizes[i] = full_img.size();

		if (work_megapix < 0) {
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else {
			if (!is_work_scale_set) {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
		}

        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        (*finder)(img, features[i]);
        features[i].img_idx = i;

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
	}

    finder->collectGarbage();
    full_img.release();
    img.release();

    return features;
}

// Find identifying features between two images using the
// SURF algorithm
//
// Pre: features is not null
//
// @arg features: 	Vector of structures containing image keypoints and
//				  	descriptors.
// @arg match_conf: Confidence two images are from the same panorama. 
//					Choose default to be 0.3
//
// Returns: Vector of structures containing information about matches 
//			between two images. It’s assumed that there is a homography 
//			between those images.
//
vector<MatchesInfo> pairwiseMatch(vector<ImageFeatures> features,
									float match_conf) {
	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(false, match_conf);

	matcher(features, pairwise_matches);
	matcher.collectGarbage();

	return pairwise_matches;
}

// Find identifying features between two images using the
// SURF algorithm
//
// Pre: features is not null, pairwise_matches is not null
//
// @arg features: Vector of structures containing image keypoints and
//				  descriptors
// @arg imgs:	  vector of Mat images
// @arg num_images: number of images to stitch
// @arg pairwise_matches: Vector of structures containing information 
//						  about matches between two images.
// @arg conf_thresh: Confidence two images are from the same panorama. 
//					 Choose default to be 1.f
//
// Returns:
//
void pruneMatches(vector<ImageFeatures> features, vector<Mat> &imgs, 
							int &num_images, vector<MatchesInfo> pairwise_matches,
							float conf_thresh) {
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches,
													conf_thresh);
	vector<Mat> img_subset;
	vector<string> img_names_subset;
	vector<Size> full_img_sizes_subset;

	for (size_t i = 0; i < indices.size(); ++i) {
        // img_names_subset.push_back(imgs[indices[i]]);
        img_subset.push_back(imgs[indices[i]]);
        // full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    num_images = static_cast<int>(img_subset.size());

    if (num_images < 2) {
    	cout << "Pruning left only 1 image to stitch." << endl;
    }

    imgs = img_subset;
}

// Describes camera parameters
//
// Pre: features is not null, pairwise_matches is not null
//
// @arg features: Vector of structures containing image keypoints and
//				  descriptors
// @arg pairwise_matches: Vector of structures containing information 
//						  about matches between two images.
// @arg ba_cost_func: Must be "reproj" or "ray". Use default value of "ray".
// @arg ba_refine_mask: Refinement mask for bundle adjustment. Use default 
//						value of "xxxxx"
// @arg conf_thresh: Confidence two images are from the same panorama. 
//					 Choose default to be 1.f
// @arg wave_correct_type: Choose "no", "horiz", "vert". Default "horiz"
// @arg warped_image_scale: No value needed.
//
// Returns: vector of camera parameters.
//
vector<CameraParams> estimate(vector<ImageFeatures> features,
				vector<MatchesInfo> pairwise_matches, string ba_cost_func,
				string ba_refine_mask, float conf_thresh,
				string wave_correct_type, float &warped_image_scale) {
	
	HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
    else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
    
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(features, pairwise_matches, cameras);

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
        focals.push_back(cameras[i].focal);

    sort(focals.begin(), focals.end());

    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (wave_correct_type != "no") {
    	WaveCorrectKind wave_correct;

    	if (wave_correct_type == "vert")
    		wave_correct = detail::WAVE_CORRECT_VERT;
    	else
    		wave_correct = detail::WAVE_CORRECT_HORIZ;

    	vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    return cameras;
}

// Warps images to fit together
//
// Pre: imgs is not null, cameras is not null
//
// @arg imgs: the images to stitch
// @arg num_of_images: number of images to stitch
// @arg cameras: vector of camera paremeters
// @arg warp_type: choose default to be "sperical"
// @arg warped_imaged_scale: 
// @arg seam_work_aspect:
// @arg expos_comp_type: Choose "no", "gain", or "gain_blocks". Default "gain_blocks"
// @arg seam_find_type:  Choose "no", "voronoi", "gc_color", "gc_colorgrad",
//						 "dp_color", "dp_colorgrad". Default "gc_color"
//
//
void warpAndCompensate(vector<Mat> &imgs, Vector<CameraParams> cameras,
					int num_images, string warp_type,
					float warped_image_scale, double seam_work_aspect,
					string expos_comp_type, string seam_find_type) {
	vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(imgs[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    Ptr<WarperCreator> warper_creator;

    if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
    else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
    else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
    else if (warp_type == "fisheye") warper_creator = new cv::FisheyeWarper();
    else if (warp_type == "stereographic") warper_creator = new cv::StereographicWarper();
    else if (warp_type == "compressedPlaneA2B1") warper_creator = new cv::CompressedRectilinearWarper(2, 1);
    else if (warp_type == "compressedPlaneA1.5B1") warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
    else if (warp_type == "compressedPlanePortraitA2B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
    else if (warp_type == "compressedPlanePortraitA1.5B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
    else if (warp_type == "paniniA2B1") warper_creator = new cv::PaniniWarper(2, 1);
    else if (warp_type == "paniniA1.5B1") warper_creator = new cv::PaniniWarper(1.5, 1);
    else if (warp_type == "paniniPortraitA2B1") warper_creator = new cv::PaniniPortraitWarper(2, 1);
    else if (warp_type == "paniniPortraitA1.5B1") warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
    else if (warp_type == "mercator") warper_creator = new cv::MercatorWarper();
    else if (warp_type == "transverseMercator") warper_creator = new cv::TransverseMercatorWarper();

    if (warper_creator.empty())
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
    }

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Mat> images_warped_f(num_images);

    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

	Ptr<ExposureCompensator> compensator;

	if (expos_comp_type == "no")
		compensator = ExposureCompensator::createDefault(ExposureCompensator::NO);
	if (expos_comp_type == "gain")
		compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	if (expos_comp_type == "gain_blocks") {
		compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	}

    compensator->feed(corners, images_warped, masks_warped);

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = new detail::NoSeamFinder();
    else if (seam_find_type == "voronoi")
        seam_finder = new detail::VoronoiSeamFinder();
    else if (seam_find_type == "gc_color")
        seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
    else if (seam_find_type == "gc_colorgrad")
        seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    else if (seam_find_type == "dp_color")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
    if (seam_finder.empty())
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unsused memory
    imgs.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();
}

int main(int argc, char const *argv[])
{
	vector<Mat> images;
    images.push_back(imread("test0.png", CV_LOAD_IMAGE_COLOR));
    images.push_back(imread("test1.png", CV_LOAD_IMAGE_COLOR));

    int num_images = static_cast<int>(images.size());
    double seam_work_aspect = 1;
    float warped_image_scale;

    vector<ImageFeatures> features = findFeatures(images, num_images, 0.6, 0.1, seam_work_aspect);
    vector<MatchesInfo> pairwise_matches = pairwiseMatch(features, 0.3f);
    pruneMatches(features, images, num_images, pairwise_matches, 1.f);
	vector<CameraParams> cameras = estimate(features, pairwise_matches, "ray",
		"xxxxx", 1.f, "horiz", warped_image_scale);

	warpAndCompensate(images, cameras, num_images,
		"plane", warped_image_scale, seam_work_aspect, "gain",
		"voronoi");

	return 0;
}