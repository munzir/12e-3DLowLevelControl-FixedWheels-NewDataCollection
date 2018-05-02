/*
 * Copyright (c) 2014-2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2014-2017, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016-2017, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "Controller.hpp"
#include <nlopt.hpp>
#include <string>
#include <iostream>

//==========================================================================
Controller::Controller(dart::dynamics::SkeletonPtr _robot,
                       dart::dynamics::BodyNode* _LeftendEffector,
                       dart::dynamics::BodyNode* _RightendEffector)
  : mRobot(_robot),
    mLeftEndEffector(_LeftendEffector),
    mRightEndEffector(_RightendEffector)
   {
  assert(_robot != nullptr);
  assert(_LeftendEffector != nullptr);
  assert(_RightendEffector != nullptr);

  int dof = mRobot->getNumDofs();
  std::cout << "[controller] DoF: " << dof << std::endl;

  mForces.setZero(dof);
  mKp.setZero();
  mKv.setZero();
  wf = 0.933881552676;
  for (int i = 0; i < 3; ++i) {
    mKp(i, i) = 750.0;
    mKv(i, i) = 250.0;
  }

  // Remove position limits
  for (int i = 0; i < dof; ++i)
    _robot->getJoint(i)->setPositionLimitEnforced(false);

  // Set joint damping
  for (int i = 0; i < dof; ++i)
    _robot->getJoint(i)->setDampingCoefficient(0, 0.5);

  // Eigen::MatrixXd xpQ   (1, DOF);
  // Eigen::MatrixXd xpdQ  (1, DOF);
  // Eigen::MatrixXd xpddQ (1, DOF);

  // // Dump data
  DataM.open      ("./data/DataM.txt");
  DataM       << "DataM" << endl;

  DataCg.open      ("./data/DataCg.txt");
  DataCg      << "DataCg" << endl;


  DataTime.open   ("./data/DataTime.txt");
  DataTime    << "DataTime" << endl;

  DataQ.open      ("./data/DataQ.txt");
  DataQ       << "DataQ" << endl;

  DatadQ.open      ("./data/DatadQ.txt");
  DatadQ       << "DatadQ" << endl;

  DataddQ.open      ("./data/DataddQ.txt");
  DataddQ       << "DataddQ" << endl;

  DataTorque.open      ("./data/DataTorque.txt");
  DataTorque       << "DataTorque" << endl;

  TargetPosition.open      ("./data/TargetPosition.txt");
  TargetPosition       << "TargetPosition" << endl;

  com.open      ("./data/com.txt");
  com       << "com" << endl;

  x_left.open      ("./data/x_left.txt");
  x_left       << "x_left" << endl;

  x_right.open      ("./data/x_right.txt");
  x_right       << "x_right" << endl;

  coeff_error.open      ("./data/coeff_error.txt");
  coeff_error       << "coeff_error" << endl;

  mTime = 0;


  //
}

//=========================================================================
Controller::~Controller() {}
//=========================================================================
struct OptParams {
  Eigen::MatrixXd P;
  Eigen::VectorXd b;
};

//=========================================================================
void printMatrix(Eigen::MatrixXd A){
  for(int i=0; i<A.rows(); i++){
    for(int j=0; j<A.cols(); j++){
      std::cout << A(i,j) << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

//========================================================================
double optFunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  OptParams* optParams = reinterpret_cast<OptParams *>(my_func_data);
  Eigen::Matrix<double, 18, 1> X(x.data());

  if (!grad.empty()) {
    Eigen::Matrix<double, 18, 1> mGrad = optParams->P.transpose()*(optParams->P*X - optParams->b);
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
  }
  return (0.5 * pow((optParams->P*X - optParams->b).norm(), 2));
}

//==============================================================================
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

//==================================================================================================
Eigen::MatrixXd error(Eigen::VectorXd dq, const int dof) {
  Eigen::MatrixXd err(DOF,1);
  for (int i = 0; i < dof; i++) {
    err(i) =  -(2/( 1 + exp(-2*dq(i)) )-1) ;
  }
  return err;
}

//=========================================================================
void Controller::update(const Eigen::Vector3d& _targetPosition) {

  mTime += 0.001;

  using namespace dart;
  using namespace std;
  const int dof = (const int)mRobot->getNumDofs();
  Eigen::VectorXd  q    = mRobot->getPositions();                 // n x 1
  Eigen::VectorXd dq    = mRobot->getVelocities();                // n x 1
  Eigen::VectorXd ddq_data   = mRobot->getAccelerations();             // n x 1
  // Eigen::VectorXd ddqref = -mKp*(q - qref) - mKv*(dq - dqref);    // n x 1
  double weightRight = 1.0, weightLeft = 1.0, weightRegulator = 1.0, weightBalance = 3.0;
  double KpzCOM = 750.0, KvzCOM = 250.0;
  //Eigen::Transform<double, 3, Eigen::Affine> Tf = mRobot->getBodyNode("Base")->getTransform();
  //Eigen::Vector3d t = Tf.translation();
  //cout << t(0) << ", " << t(1) << ", " << t(2) << endl;

  // Left arm
  Eigen::Vector3d xLft    = mLeftEndEffector->getTransform().translation();
  Eigen::Vector3d dxLft   = mLeftEndEffector->getLinearVelocity();
  math::LinearJacobian JvLft   = mLeftEndEffector->getLinearJacobian();       // 3 x n
  math::LinearJacobian dJvLft  = mLeftEndEffector->getLinearJacobianDeriv();  // 3 x n
  Eigen::Vector3d ddxrefLft = -mKp*(xLft - _targetPosition) - mKv*dxLft;
  Eigen::Vector3d zeroColumn(0.0, 0.0, 0.0);
  Eigen::Matrix<double, 3, 7> zero7Columns;
  zero7Columns << zeroColumn, zeroColumn, zeroColumn, zeroColumn, \
              zeroColumn, zeroColumn, zeroColumn;
  Eigen::Matrix<double, 3, 18> FullJacobianLeft;
  FullJacobianLeft  << JvLft.block<3,3>(0,0), zeroColumn, JvLft.block<3,7>(0,3), zero7Columns;
  Eigen::Matrix<double, 3, 18> FullJacobianDerLft;
  FullJacobianDerLft  <<  dJvLft.block<3,3>(0,0), zeroColumn, dJvLft.block<3,7>(0,3), zero7Columns;
  Eigen::MatrixXd PLeft   = weightLeft*FullJacobianLeft;
  Eigen::VectorXd bLeft   = -weightLeft* ( FullJacobianDerLft*dq - ddxrefLft );

  // Right Arm
  Eigen::Vector3d xRgt    = mRightEndEffector->getTransform().translation();
  Eigen::Vector3d dxRgt   = mRightEndEffector->getLinearVelocity();
  math::LinearJacobian JvRgt   = mRightEndEffector->getLinearJacobian();       // 3 x n
  math::LinearJacobian dJvRgt  = mRightEndEffector->getLinearJacobianDeriv();  // 3 x n
  Eigen::Vector3d ddxrefRgt = -mKp*(xRgt - _targetPosition) - mKv*dxRgt;
  Eigen::Matrix<double, 3, 18> FullJacobianRight;
  FullJacobianRight <<  JvRgt.block<3,3>(0,0), zeroColumn, zero7Columns, JvRgt.block<3,7>(0,3);
  Eigen::Matrix<double, 3, 18> FullJacobianDerRgt;
  FullJacobianDerRgt  <<  dJvRgt.block<3,3>(0,0), zeroColumn, zero7Columns, dJvRgt.block<3,7>(0,3);
  Eigen::MatrixXd PRight  = weightRight*FullJacobianRight;
  Eigen::VectorXd bRight  = -weightRight*( FullJacobianDerRgt*dq - ddxrefRgt );

  // CoM
  double zCOM = mRobot->getCOM()(2);
  double dzCOM = mRobot->getCOMLinearVelocity()(2);
  Eigen::VectorXd JzCOM = mRobot->getCOMLinearJacobian().block<1,18>(2,0);
  Eigen::VectorXd dJzCOM = mRobot->getCOMLinearJacobianDeriv().block<1,18>(2,0);
  double ddzCOMref = -KpzCOM*zCOM - KvzCOM*dzCOM;
  Eigen::MatrixXd PBalance = weightBalance*JzCOM;
  double bBalance = -weightBalance*(dJzCOM.transpose()*dq - ddzCOMref);

  // Regulator
  Eigen::MatrixXd PRegulator = weightRegulator*Eigen::MatrixXd::Identity(dof, dof);
  Eigen::VectorXd bRegulator = -weightRegulator*10*dq;

  // Optimizer stuff
  OptParams optParams;
  Eigen::MatrixXd NewP(PRight.rows() + PLeft.rows() + PRegulator.rows() + PBalance.cols(), PRight.cols() );
  NewP << PRight,
          PLeft,
          PRegulator,
          PBalance.transpose();
  Eigen::VectorXd NewB(bRight.rows() + bLeft.rows() + bRegulator.rows() + 1, bRight.cols() );
  NewB << bRight,
          bLeft,
          bRegulator,
          bBalance;
  optParams.P = NewP;
  optParams.b = NewB;
  nlopt::opt opt(nlopt::LD_SLSQP, dof);
  std::vector<double> ddq_vec(dof);
  double minf;
  opt.set_min_objective(optFunc, &optParams);
  opt.set_xtol_rel(1e-4);
  opt.set_maxtime(0.005);
  opt.optimize(ddq_vec, minf);
  Eigen::Matrix<double, 18, 1> ddq(ddq_vec.data());
 
  // Torques
  Eigen::MatrixXd M     = mRobot->getMassMatrix();                // n x n
  Eigen::VectorXd Cg    = mRobot->getCoriolisAndGravityForces();  // n x 1
  mForces = M*ddq + Cg;
  Eigen::Matrix<double, 18, 18> errCoeff = Eigen::Matrix<double, 18, 18>::Identity();
  errCoeff(0,0) =   1.0;
  errCoeff(1,1) =   1.2;
  errCoeff(2,2) =   1.3;
  errCoeff(3,3) =   1.0;
  
  // Left Arm
  errCoeff(4,4) =   30;
  errCoeff(5,5) =   30;
  errCoeff(6,6) =   15;
  errCoeff(7,7) =   15;
  errCoeff(8,8) =   7;
  errCoeff(9,9) =   7;
  errCoeff(10,10) = 1;
  
  // Right Arm
  errCoeff(11,11) = 30;
  errCoeff(12,12) = 30;
  errCoeff(13,13) = 15;
  errCoeff(14,14) = 15;
  errCoeff(15,15) = 7;
  errCoeff(16,16) = 7;
  errCoeff(17,17) = 1;


  Eigen::VectorXd mForceErr = mForces + errCoeff*error(dq, dof);
  mRobot->setForces(mForceErr);
  DataTime    << mTime << endl;
  DataQ       << q.transpose() << endl;
  DatadQ    << dq.transpose() << endl;
  DataddQ   << ddq_data.transpose() << endl;
  DataM   << M << endl;
  DataCg  << Cg << endl;
  DataTorque << mForces.transpose() << endl;
  TargetPosition<<_targetPosition.transpose()<<endl;
  x_left<<xLft.transpose()<<endl;
  x_right<<xRgt.transpose()<<endl;
  com << dzCOM << endl;
  coeff_error<<errCoeff<<endl;

  // Closing operation
  double T = 10;
  if (mTime >= 2*T ) {
    // cout << "Time period met. Closing simulation ..." << endl;
    cout << "Time period met. Stopping data recording ...";
    DataTime.close();
    DataQ.close();
    DatadQ.close();
    DataddQ.close();
    DataM.close();
    DataCg.close();
    DataTorque.close();
    x_left.close();
    x_right.close();
    TargetPosition.close();
    com.close();
    coeff_error.close();
    // dataQdot.close();
    // dataQdotdot.close();
    // dataTorque.close();
    // dataM.close();
    // dataCg.close();
    // dataError.close();
    cout << "File handles closed!" << endl << endl << endl;
    exit (EXIT_FAILURE);
}
}
//=========================================================================
dart::dynamics::SkeletonPtr Controller::getRobot() const {
  return mRobot;
}

//=========================================================================
dart::dynamics::BodyNode* Controller::getEndEffector(const std::string &s) const {
  if (s.compare("left")) {  return mLeftEndEffector; }
  else if (s.compare("right")) { return mRightEndEffector; }
}

//=========================================================================
void Controller::keyboard(unsigned char /*_key*/, int /*_x*/, int /*_y*/) {
}
