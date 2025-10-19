#include "TRandom2.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TGClient.h"
#include "TStyle.h"


#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompLU.h"
#include "TGraph.h"
#include "TLegend.h"

#include <vector>
#include <cmath>


#include <iostream>
using namespace std;

using TMath::Log;

//parms
const double xmin=1;
const double xmax=20;
const int npoints=12;
const double sigma=0.2;

double f(double x){
  const double a=0.5;
  const double b=1.3;
  const double c=0.5;
  return a+b*Log(x)+c*Log(x)*Log(x);
}

void getX(double *x){
  double step=(xmax-xmin)/npoints;
  for (int i=0; i<npoints; i++){
    x[i]=xmin+i*step;
  }
}

void getY(const double *x, double *y, double *ey){
  static TRandom2 tr(0);
  for (int i=0; i<npoints; i++){
    y[i]=f(x[i])+tr.Gaus(0,sigma);
    ey[i]=sigma;
  }
}


void leastsq(){
  double x[npoints];
  double y[npoints];
  double ey[npoints];
  getX(x);
  getY(x,y,ey);
  auto tg = new TGraphErrors(npoints,x,y,0,ey);
  tg->Draw("alp");
}

double f_model(double x, double a, double b, double c){
  const double lx = Log(x);   
  return a + b*lx + c*lx*lx;
}

struct FitResult {
  double a,b,c;
  double chi2;
  int dof; 
};

FitResult wls_fit(const double* x, const double* y, const double* ey, int n) {
  // define X matrix 
  TMatrixD X(n, 3);
  TVectorD Y(n), W(n); 

  for (int i = 0; i < n; ++i) {
    const double lx = Log(x[i]);
    X(i, 0) = 1;
    X(i, 1) = lx; 
    X(i, 2) = lx*lx; 
    Y[i] = y[i]; 
    W[i] = 1/(ey[i]*ey[i]);
  }

  // define normal equation 
  TMatrixD XT(3, n); XT.Transpose(X);

    // XT_W = XT * W (W is diagonal → scale each column by W[i])
  TMatrixD XT_W(3, n);
  for (int r=0; r<3; ++r)
    for (int c=0; c<n; ++c)
      XT_W(r,c) = XT(r,c) * W[c];

  TMatrixD A(3,3); A.Mult(XT_W, X);     // A = X^T W X
  TVectorD b(3);   b = XT_W * Y;        // b = X^T W y

  // Solve for parameters p
  TDecompLU lu(A);
  TVectorD p = b;
  if (!lu.Solve(p)) {
    std::cerr << "wls_fit: linear solve failed\n";
    return FitResult{0,0,0,0.0, n-3};
  }

  // Compute chi^2
  double chi2 = 0.0;
  for (int i = 0; i < n; ++i){
    const double lx   = X(i,1);
    const double lx2  = X(i,2);
    const double yhat = p[0] + p[1]*lx + p[2]*lx2;
    const double pull = (Y[i] - yhat) / std::sqrt(1.0/W[i]); // = (Y - yhat)/ey
    chi2 += pull*pull;
  }

  return { p[0], p[1], p[2], chi2, n - 3 };
}

int main(int argc, char **argv){
  TApplication theApp("App", &argc, argv); // init ROOT App for displays

  // ******************************************************************************
  // ** this block is useful for supporting both high and std resolution screens **
  UInt_t dh = gClient->GetDisplayHeight()/2;   // fix plot to 1/2 screen height  
  //UInt_t dw = gClient->GetDisplayWidth();
  UInt_t dw = 1.1*dh;
  // ******************************************************************************

  gStyle->SetOptStat(0); // turn off histogram stats box


  TCanvas *tc = new TCanvas("c1","Sample dataset",dw,dh);

  double lx[npoints];
  double ly[npoints];
  double ley[npoints];

  getX(lx);
  getY(lx,ly,ley);
  auto tgl = new TGraphErrors(npoints,lx,ly,0,ley);
  tgl->SetTitle("Pseudoexperiment;x;y");
  
  // An example of one pseudo experiment
  tgl->Draw("alp");
  tc->Draw();

  
  // *** modify and add your code here ***
  FitResult demo = wls_fit(lx, ly, ley, npoints);

  const int Ncurv = 400;
  std::vector<double> xs(Ncurv), yfit(Ncurv), ytru(Ncurv);
  for (int i=0; i<Ncurv; ++i){
    xs[i] = xmin + (xmax - xmin) * (double)i / (Ncurv-1);
    yfit[i] = f_model(xs[i], demo.a, demo.b, demo.c);
    ytru[i] = f_model(xs[i], /*a_true*/0.5, /*b_true*/1.3, /*c_true*/0.5);
  }
  auto grFit = new TGraph(Ncurv, xs.data(), yfit.data()); grFit->SetLineWidth(2);
  auto grTru = new TGraph(Ncurv, xs.data(), ytru.data()); grTru->SetLineStyle(2);

  tgl->SetTitle(Form("Pseudoexperiment; x; y  (chi2/ndf = %.2f)", demo.chi2/(double)demo.dof));
  grFit->Draw("L SAME");
  grTru->Draw("L SAME");
  auto leg = new TLegend(0.60,0.75,0.88,0.90);
  leg->AddEntry(grFit, "WLS fit","l");
  leg->AddEntry(grTru, "Truth",  "l");
  leg->Draw();
  tc->Modified(); tc->Update();

  // ---------- Study over many pseudo-experiments ----------
  const int nexperiments = (int) 1e6;

  // Choose display ranges near the truth (improves the 2D density look)
  TH2F *h1 = new TH2F("h1","Parameter b vs a; a; b", 80, 0.2, 0.8, 80, 0.7, 1.9);
  TH2F *h2 = new TH2F("h2","Parameter c vs a; a; c", 80, 0.2, 0.8, 80, 0.2, 0.9);
  TH2F *h3 = new TH2F("h3","Parameter c vs b; b; c", 80, 0.7, 1.9, 80, 0.2, 0.9);
  TH1F *h4 = new TH1F("h4","Reduced #chi^{2};;Density", 100, 0.0, 2.5);
  h4->Sumw2();

  // Reuse x positions but regenerate y each time
  double xbuf[npoints];
  getX(xbuf);

  for (int ie = 0; ie < nexperiments; ++ie){
    double ybuf[npoints], eybuf[npoints];
    getY(xbuf, ybuf, eybuf);

    FitResult r = wls_fit(xbuf, ybuf, eybuf, npoints);

    h1->Fill(r.a, r.b);
    h2->Fill(r.a, r.c);
    h3->Fill(r.b, r.c);
    h4->Fill(r.chi2 / r.dof, 1.0 / nexperiments); // scale so it looks like a PDF
  }

  // ---------- Draw results ----------
  TCanvas *tc2 = new TCanvas("c2","my study results",200,200,dw,dh);
  tc2->Divide(2,2);
  tc2->cd(1); gPad->SetRightMargin(0.14); h1->Draw("COLZ");
  tc2->cd(2); gPad->SetRightMargin(0.14); h2->Draw("COLZ");
  tc2->cd(3); gPad->SetRightMargin(0.14); h3->Draw("COLZ");
  tc2->cd(4); h4->Draw("HIST");
  tc2->Modified(); tc2->Update();

  TString outname = "LSQFit_cpp.pdf";

  // open PDF file (first page)
  tc->Print(outname + "(", "pdf");   // "(" means start multipage PDF
  tc2->Print(outname + ")", "pdf");  // ")" closes it

  cout << "Saved results to " << outname << endl;

  // **************************************
  
  // cout << "Press ^c to exit" << endl;
  // theApp.SetIdleTimer(30,".q");  // set up a failsafe timer to end the program  
  // theApp.Run();
}
