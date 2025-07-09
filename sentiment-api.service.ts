import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SentimentApiService {
  private apiUrl = 'http://localhost:8000'; // Update if backend is hosted elsewhere

  constructor(private http: HttpClient) {}

  getAllStocksSentiment(time: string = '24h'): Observable<any> {
    return this.http.get(`${this.apiUrl}/all_stocks_sentiment`, {
      params: new HttpParams().set('time', time)
    });
  }

  getStockSentiment(stock: string, time: string = '24h'): Observable<any> {
    return this.http.get(`${this.apiUrl}/get_sentiment_score`, {
      params: new HttpParams().set('stock', stock).set('time', time)
    });
  }
} 