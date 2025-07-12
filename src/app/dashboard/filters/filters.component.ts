import { Component, EventEmitter, Output } from '@angular/core';

@Component({
  selector: 'app-filters',
  templateUrl: './filters.component.html',
  styleUrls: ['./filters.component.scss']
})
export class FiltersComponent {
  @Output() timeframeChange = new EventEmitter<string>();
  @Output() sentimentChange = new EventEmitter<string>();
  @Output() stockChange = new EventEmitter<string>();

  timeframes = [
    { label: '24 Hours', value: '24h' },
    { label: '7 Days', value: '7d' },
    { label: '30 Days', value: '30d' }
  ];
  sentiments = [
    { label: 'All', value: '' },
    { label: 'Bullish', value: 'positive' },
    { label: 'Bearish', value: 'negative' },
    { label: 'Neutral', value: 'neutral' }
  ];
  stocks = [
    { label: 'All', value: '' },
    { label: 'RELIANCE', value: 'RELIANCE' },
    { label: 'TCS', value: 'TCS' },
    { label: 'INFY', value: 'INFY' },
    { label: 'HDFCBANK', value: 'HDFCBANK' },
    { label: 'ICICIBANK', value: 'ICICIBANK' }
  ];

  selectedTimeframe = '24h';
  selectedSentiment = '';
  selectedStock = '';

  onTimeframeChange(value: string) {
    this.selectedTimeframe = value;
    this.timeframeChange.emit(value);
  }
  onSentimentChange(value: string) {
    this.selectedSentiment = value;
    this.sentimentChange.emit(value);
  }
  onStockChange(value: string) {
    this.selectedStock = value;
    this.stockChange.emit(value);
  }
} 