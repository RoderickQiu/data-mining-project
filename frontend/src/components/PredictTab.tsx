import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { apiService, PredictRequest } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2 } from 'lucide-react';

const PredictTab: React.FC = () => {
    const [inputIds, setInputIds] = useState<string>('');
    const [inputRtime, setInputRtime] = useState<string>('');
    const [inputCat, setInputCat] = useState<string>('');
    const [predictions, setPredictions] = useState<number[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string>('');

    const handlePredict = async () => {
        try {
            setLoading(true);
            setError('');

            // Parse input strings to arrays
            const ids = inputIds.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            const rtime = inputRtime.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            const cat = inputCat.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

            if (ids.length === 0 || rtime.length === 0 || cat.length === 0) {
                throw new Error('请输入有效的数字序列');
            }

            if (ids.length !== rtime.length || ids.length !== cat.length) {
                throw new Error('所有输入序列的长度必须相同');
            }

            const request: PredictRequest = {
                input_ids: ids,
                input_rtime: rtime,
                input_cat: cat,
            };

            const response = await apiService.predict(request);
            setPredictions(response.probs.slice(0, ids.length));
        } catch (err) {
            setError(err instanceof Error ? err.message : '预测失败');
        } finally {
            setLoading(false);
        }
    };

    const chartData = predictions.map((prob, index) => ({
        index: index + 1,
        probability: prob,
    }));

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>模型预测</CardTitle>
                    <CardDescription>
                        输入学习序列数据，获取模型预测结果
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="input-ids">题目ID序列 (用逗号分隔)</Label>
                        <Input
                            id="input-ids"
                            placeholder="例如: 1,2,3,4,5"
                            value={inputIds}
                            onChange={(e) => setInputIds(e.target.value)}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="input-rtime">响应时间序列 (用逗号分隔)</Label>
                        <Input
                            id="input-rtime"
                            placeholder="例如: 1000,2000,1500,3000,2500"
                            value={inputRtime}
                            onChange={(e) => setInputRtime(e.target.value)}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="input-cat">题目类别序列 (用逗号分隔)</Label>
                        <Input
                            id="input-cat"
                            placeholder="例如: 1,2,1,3,2"
                            value={inputCat}
                            onChange={(e) => setInputCat(e.target.value)}
                        />
                    </div>

                    {error && (
                        <div className="text-red-500 text-sm">{error}</div>
                    )}

                    <Button onClick={handlePredict} disabled={loading} className="w-full">
                        {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                        {loading ? '预测中...' : '开始预测'}
                    </Button>
                </CardContent>
            </Card>

            {predictions.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle>预测结果</CardTitle>
                        <CardDescription>
                            模型对每个题目的预测概率
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="index" />
                                    <YAxis domain={[0, 1]} />
                                    <Tooltip
                                        formatter={(value: number) => [value.toFixed(4), '预测概率']}
                                        labelFormatter={(label) => `题目 ${label}`}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="probability"
                                        stroke="#8884d8"
                                        strokeWidth={2}
                                        dot={{ r: 4 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
                            {predictions.map((prob, index) => (
                                <div key={index} className="text-center p-2 bg-muted rounded">
                                    <div className="text-sm text-muted-foreground">题目 {index + 1}</div>
                                    <div className="font-semibold">{prob.toFixed(4)}</div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
};

export default PredictTab; 