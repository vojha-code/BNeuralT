package reporting;

import dataCSVReader.DataPreporcessor;

public class RegressionStat {

	public double mse; //mean squared error
	public double rmse;
	public double mae;

	public double rxy;//r Correlation Coefficient
	public double r2; // r2 - Coefficient of determination Nash–Sutcliffe model efficiency coefficient



	public RegressionStat(double[][] targets, double[][] prediction, double[] mTargetMin, double[] mTargetMax, double[] m_scale) {
		//Return MSE/R^2
		//System.out.println("MSE");
		int m_target_attr_count = targets[0].length;
		if (m_target_attr_count == 1) {
			int length = targets.length;
			//check de-normalised
			double[] x = new double[length];
			double[] y = new double[length];
			double xMean = 0.0, yMean = 0.0;
			double ssReg = 0.0;
			//double ssRegABS = 0.0;
			DataPreporcessor dp = new DataPreporcessor();
			for (int i = 0; i < length; i++) {
				x[i] = dp.denormalize(targets[i][0], mTargetMin[0], mTargetMax[0], m_scale[0], m_scale[1]);
				y[i] = dp.denormalize(prediction[i][0], mTargetMin[0], mTargetMax[0], m_scale[0], m_scale[1]);
				//System.out.println(m_data_target_TO_tree[i][0] + " = "+x[i] +" ,"+ m_data_prediction_OF_tree[i][0]+ " = "+y[i]);
				//System.out.println(x[i] + "," + y[i]);
				xMean = xMean + x[i]; //x
				yMean = yMean + y[i]; //y
				ssReg = ssReg + Math.pow((x[i] - y[i]), 2);
				//ssRegABS = ssRegABS + Math.abs(x[i] - y[i]);
			}
			xMean = xMean / length;
			yMean = yMean / length;

			double aSqSum = 0.0;//ssTotal
			double bSqSum = 0.0;
			double axbSUM = 0.0;

			for (int i = 0; i < length; i++) {
				double a = x[i] - xMean;
				double b = y[i] - yMean;
				aSqSum = aSqSum + a * a;
				bSqSum = bSqSum + b * b;
				axbSUM = axbSUM + a * b;
			}
			mse = ssReg / length;
			//rmse = Math.sqrt(mse);
			//mae = ssRegABS / length;
			rxy = axbSUM / Math.sqrt(aSqSum * bSqSum);// sum(xi-mux)2*sum(yi-muy)2 / sqrt(sum(xi-mux)2*sum(yi-muy)2)
			r2 = 1.0 - (ssReg / aSqSum); //1 - ssREG/ssTotal /
		} else {
			System.out.println("truth class " + m_target_attr_count + "more than necessary for regression");
			System.exit(0);
		}
	}

	public RegressionStat(double[][] targets, double[][] prediction) {
		//Return MSE/R^2
		//System.out.println("MSE");
		int m_target_attr_count = targets[0].length;
		if (m_target_attr_count == 1) {
			int length = targets.length;
			//check de-normalised
			double[] x = new double[length];
			double[] y = new double[length];
			double xMean = 0.0, yMean = 0.0;
			double ssReg = 0.0;
			//double ssRegABS = 0.0;
			for (int i = 0; i < length; i++) {
				x[i] = targets[i][0];
				y[i] = prediction[i][0];
				//System.out.println(m_data_target_TO_tree[i][0] + " = "+x[i] +" ,"+ m_data_prediction_OF_tree[i][0]+ " = "+y[i]);
				//System.out.println(x[i] + "," + y[i]);
				xMean = xMean + x[i]; //x
				yMean = yMean + y[i]; //y
				ssReg = ssReg + Math.pow((x[i] - y[i]), 2);
				//ssRegABS = ssRegABS + Math.abs(x[i] - y[i]);
			}
			xMean = xMean / length;
			yMean = yMean / length;

			double aSqSum = 0.0;//ssTotal
			double bSqSum = 0.0;
			double axbSUM = 0.0;

			for (int i = 0; i < length; i++) {
				double a = x[i] - xMean;
				double b = y[i] - yMean;
				aSqSum = aSqSum + a * a;
				bSqSum = bSqSum + b * b;
				axbSUM = axbSUM + a * b;
			}
			mse = ssReg / length;
			//rmse = Math.sqrt(mse);
			//mae = ssRegABS / length;
			rxy = axbSUM / Math.sqrt(aSqSum * bSqSum);// sum(xi-mux)2*sum(yi-muy)2 / sqrt(sum(xi-mux)2*sum(yi-muy)2)
			r2 = 1.0 - (ssReg / aSqSum); //1 - ssREG/ssTotal /
		} else {
			System.out.println("truth class " + m_target_attr_count + "more than necessary for regression");
			System.exit(0);
		}
	}


	public RegressionStat(double[] x, double[] y) {
		//Return MSE/R^2
		//System.out.println("MSE");
		int m_target_attr_count = 1;

		int length = x.length;
		//check de-normalised
		double xMean = 0.0, yMean = 0.0;
		double ssReg = 0.0;
		//double ssRegABS = 0.0;
		for (int i = 0; i < length; i++) {
			//System.out.println(m_data_target_TO_tree[i][0] + " = "+x[i] +" ,"+ m_data_prediction_OF_tree[i][0]+ " = "+y[i]);
			//System.out.println(x[i] + "," + y[i]);
			xMean = xMean + x[i]; //x
			yMean = yMean + y[i]; //y
			ssReg = ssReg + Math.pow((x[i] - y[i]), 2);
			//ssRegABS = ssRegABS + Math.abs(x[i] - y[i]);
		}
		xMean = xMean / length;
		yMean = yMean / length;

		double aSqSum = 0.0;//ssTotal
		double bSqSum = 0.0;
		double axbSUM = 0.0;

		for (int i = 0; i < length; i++) {
			double a = x[i] - xMean;
			double b = y[i] - yMean;
			aSqSum = aSqSum + a * a;
			bSqSum = bSqSum + b * b;
			axbSUM = axbSUM + a * b;
		}
		mse = ssReg / length;
		//rmse = Math.sqrt(mse);
		//mae = ssRegABS / length;
		rxy = axbSUM / Math.sqrt(aSqSum * bSqSum);// sum(xi-mux)2*sum(yi-muy)2 / sqrt(sum(xi-mux)2*sum(yi-muy)2)
		r2 = 1.0 - (ssReg / aSqSum); //1 - ssREG/ssTotal /
	}

}
